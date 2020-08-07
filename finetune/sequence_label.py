#coding:utf8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


import os
import time
import argparse
import numpy as np
import multiprocessing
import sys

import paddle
import logging
import paddle.fluid as fluid

from six.moves import xrange

from model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    src_ids = fluid.layers.data(name='input_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    sent_ids = fluid.layers.data(name='sent_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    pos_ids = fluid.layers.data(name='pos_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    # 这个好像没用...
    task_ids = fluid.layers.data(name='task_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    input_mask = fluid.layers.data(name='input_mask', shape=[-1, args.max_seq_len, 1], dtype='float32')
    labels = fluid.layers.data(name='labels', shape=[-1, args.max_seq_len, 1], dtype='int64')
    seq_lens = fluid.layers.data(name='seq_lens', shape=[-1], dtype='int64')

    # 该方法会返回一个 DataLoader 对象
    # feed_list 参数： python list|tuple of Variables ，列表元素都由 fluid.layers.data() 创建
    # capacity 参数：  DataLoader 对象内部维护队列的容量大小。单位是 batch 数量。若 reader 读取速度较快，建议设置较大的 capacity
    # iterable 参数：  所创建的 DataLoader 对象是否可迭代，默认参数是 True
    pyreader = fluid.io.DataLoader.from_generator(feed_list=[src_ids, sent_ids, pos_ids, task_ids, input_mask, labels, seq_lens], 
            capacity=70,
            iterable=False)

    # ernie 对象会创建 baseline 使用的基于 transformer 的 encoder-decoder 模型
    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    # enc_out shape=[]
    # enc_out 是 encoder 的输出
    enc_out = ernie.get_sequence_output()
    # Dropout是一种正则化手段，通过在训练过程中阻止神经元节点间的相关性来减少过拟合。
    # 根据给定的丢弃概率，dropout操作符按丢弃概率随机将一些神经元输出设置为0，其他的仍保持不变。
    # upscale_in_train 的实现原理：
    # - train: out = input * mask / ( 1.0 - dropout_prob )
    # - inference: out = input
    enc_out = fluid.layers.dropout(
        x=enc_out, dropout_prob=0.1, dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        name="logits",
        input=enc_out,
        # 输出 size 为 num_labels=7
        size=args.num_labels,
        # num_flatten_dims 表示 fc 层输出结果保留的维度，
        # eg. 如果输入 shape=[2,3,4,5] & num_flatten_dims=2 -> output shape=[2,3,size]
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b",
            initializer=fluid.initializer.Constant(0.)))
    # logits 维度应该是3，dim=0 batch_size dim=1 max_seq_len dim=2 7
    infers = fluid.layers.argmax(logits, axis=2)
    # feed_targets_name 在 train.py 里面设置为 False
    # feed_targets_name 在 save_model.py 里面设置为 True ， 用于 save_inference_model
    #  predictor.py 中使用 load_inference_model 读取前一步保存的模型， 因此可以认为 feed_targets_name 主要在预测阶段使用
    if is_prediction:
        feed_targets_name = [
            src_ids.name, sent_ids.name,  pos_ids.name,input_mask.name
        ]
    else:
        feed_targets_name = []
    # ret_infers 把一个 batch 所有 tokens 拉到一起
    ret_infers = fluid.layers.reshape(x=infers, shape=[-1, 1])
    # 按照 seq_lens 的长度来去掉 labels infers 中的padding
    lod_labels = fluid.layers.sequence_unpad(labels, seq_lens)
    lod_infers = fluid.layers.sequence_unpad(infers, seq_lens)

    # 该接口用来计算语块识别（chunk detection）的准确率、召回率和F1值，
    # 常用于 NER 等序列标注任务中。
    # 返回（按照顺序依次是）：
    # 准确率、召回率、F1值，识别出的语块数目、标签中的语块数目、正确识别的语块数目
    # 返回值都 Tensor ，准确率、召回率、F1值的数据类型为 float32 ，其他的数据类型为 int64 。
    (_, _, _, num_infer, num_label, num_correct) = fluid.layers.chunk_eval(
         input=lod_infers,
         label=lod_labels,
         # baseline 使用的是 IOB 结构
         chunk_scheme=args.chunk_scheme,
         # num_chunk_types 表示标签中的语块类型数
         num_chunk_types=((args.num_labels-1)//(len(args.chunk_scheme)-1)))

    # fluid.layers.flatten 把输入 tensor 转换成 2-D tensor ， 
    # axis=2 代表把 axis<2 的 flatten 成 0维， axis>=2 的 flatten 成 1维
    # 这里的 flatten 操作应该是把 一个 batch 的 token 平铺到一起了
    labels = fluid.layers.flatten(labels, axis=2)
    # 使用 logits 和 labels 来计算交叉熵
    # 序列标注本质上还是一个分类问题
    # 这里因为 return_softmax=True ，所以返回 ce_loss 是交叉熵损失函数
    # probs 是softmax归一化后的 logits
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=fluid.layers.flatten(
            logits, axis=2),
        label=labels,
        return_softmax=True)

    # input_mask 决定了哪些地方需要计算loss ， 哪些位置不需要计算 loss
    input_mask = fluid.layers.flatten(input_mask, axis=2)
    ce_loss = ce_loss * input_mask
    # mean 返回所有标量 loss ， 即所有 token 上 loss 的平均值
    loss = fluid.layers.mean(x=ce_loss)
    if not is_prediction:
        graph_vars = {
            "logits":logits,
            "infers":infers,
            "loss": loss,
            "probs": probs,
            "seqlen": seq_lens,
            "num_infer": num_infer,
            "num_label": num_label,
            "num_correct": num_correct,
        }
    else:
        # predictor.py 里，模型只返回 infers
        graph_vars = {
            "probs": infers,
        }
    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars,feed_targets_name


def calculate_f1(num_label, num_infer, num_correct):

    num_infer = np.sum(num_infer)
    num_label = np.sum(num_label)
    num_correct = np.sum(num_correct)
    
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(exe,
             program,
             pyreader,
             graph_vars,
             tag_num,
             dev_count=1):
    fetch_list = [
        graph_vars["num_infer"].name, graph_vars["num_label"].name,
        graph_vars["num_correct"].name
    ]

    total_label, total_infer, total_correct = 0.0, 0.0, 0.0
    time_begin = time.time()
    pyreader.start()
    while True:
        try:
            np_num_infer, np_num_label, np_num_correct = exe.run(program=program,
                                                    fetch_list=fetch_list)
            total_infer += np.sum(np_num_infer)
            total_label += np.sum(np_num_label)
            total_correct += np.sum(np_num_correct)

        except fluid.core.EOFException:
            pyreader.reset()
            break

    precision, recall, f1 = calculate_f1(total_label, total_infer,
                                         total_correct)
    time_end = time.time()
    return  \
        "[evaluation] f1: %f, precision: %f, recall: %f, elapsed time: %f s" \
        % (f1, precision, recall, time_end - time_begin)


def chunk_predict( np_probs, np_lens, dev_count=1):
    #inputs = np_inputs.reshape([-1]).astype(np.int32)
    probs = np_probs.reshape([-1, np_probs.shape[-1]])

    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    out = []
    for dev_index in xrange(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in xrange(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            prob = probs[seq_st:seq_en, :]
            infers = np.argmax(prob, -1)
            out.append((
                    #inputs[seq_st:seq_en].tolist(),
                    infers.tolist(),
                    prob.tolist()))
        base_index += max_len * len(lens)
    return out


def predict(exe,
            test_program,
            test_pyreader,
            graph_vars,
            dev_count=1):
    fetch_list = [
    #    graph_vars["inputs"].name,
        graph_vars["probs"].name,
        graph_vars["seqlen"].name,
    ]

    test_pyreader.start()
    res = []
    while True:
        try:
            probs, np_lens = exe.run(program=test_program,
                                        fetch_list=fetch_list)
            r = chunk_predict( probs, np_lens, dev_count)
            res += r
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    return res

