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
import logging
import multiprocessing
import sys
import json
import re
sys.path.append(".")
sys.path.append("..")
# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid
from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
from reader.task_reader import SequenceLabelReaderWithPremise
from model.ernie import ErnieConfig
from finetune.sequence_label import create_model

from utils.args import print_arguments, check_cuda, prepare_logger, ArgumentGroup
from utils.init import init_pretraining_params,init_checkpoint
from finetune_args import parser

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
pwd_path = os.path.abspath(os.path.dirname(__file__))
model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
model_g.add_arg("ernie_config_path",            str,  os.path.join(pwd_path,"ERNIE_stable-1.0.1/ernie_config.json"),  "Path to the json file for ernie model config.")
model_g.add_arg("init_checkpoint",              str,  os.path.join(pwd_path,"inferences"),  "Init checkpoint to resume training from.")
model_g.add_arg("save_inference_model_path",    str,  os.path.join(pwd_path,"inferences"),"If set, save the inference model to this path.")
model_g.add_arg("use_fp16",                     bool, False, "Whether to resume parameters from fp16 checkpoint.")
model_g.add_arg("num_labels",                   int,  7,     "num labels for classify")
model_g.add_arg("chunk_scheme",                  str,  "IOB", "Set for sequence to chunk.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("vocab_path",          str,  os.path.join(pwd_path,"ERNIE_stable-1.0.1/vocab.txt"),  "Vocabulary path.")
data_g.add_arg("label_map_config",    str,   os.path.join(pwd_path,"data/label_map.json"),  "Label_map_config json file.")
data_g.add_arg("max_seq_len",         int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  8,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",          bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("do_prediction",     bool,   True,  "Whether to do prediction on test set.")

args = parser.parse_args()

# yapf: enable.
class Predictor():
    def __init__(self):
        self.args = args

    def predict(self, records):
        
        if self.args.use_cuda:
            # 该接口根据 device_ids 创建一个或多个 fluid.CUDAPlace 对象，并返回所创建的对象列表
            # fluid.CUDAPlace 是GPU设备描述符， fluid.CUDAPlace的设备id表示GPU编号，从0开始
            # 这里的 place 变量，在后面创建exe的时候会用到
            dev_list = fluid.cuda_places()
            place = dev_list[0]
            # 如果 dev_count>1 ，此时多卡训练需要相应地分配mini-batch数据到每个GPU
            dev_count = len(dev_list)
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        reader = SequenceLabelReaderWithPremise(
            vocab_path=self.args.vocab_path,
            label_map_config=self.args.label_map_config,
            max_seq_len=self.args.max_seq_len,
            do_lower_case=self.args.do_lower_case,
            in_tokens=False,
            is_inference=True)
        index_char = self.load_vocab()

        exe = fluid.Executor(place)
        # exe = fluid.Executor(fluid.CUDAPlace(0)) # CUDAPlace(0) CPUPlace()
        path = self.args.init_checkpoint
        [inference_program, feed_target_names, fetch_targets] =\
            fluid.io.load_inference_model(dirname=path, executor=exe,model_filename="__model__",params_filename="weights")

        predict_data_generator = reader.data_generator_json(
            input_json = records,
            batch_size = self.args.batch_size,
            epoch = 1,
            shuffle = False
            )

        total_time = 0
        res = []

        index =0
        for sample in predict_data_generator():
            index=index+1
            #this is one batch of data
            src_ids    = sample[0]
            sent_ids   = sample[1]
            pos_ids    = sample[2]
            task_ids   = sample[3]
            input_mask = sample[4]
            labels = sample[5]
            len_seqs = sample[6]
            inputs = [self.array2tensor(ndarray) for ndarray in [src_ids, sent_ids, pos_ids,input_mask]]
            begin_time = time.time()
            outputs =  exe.run(inference_program,
                        feed={feed_target_names[0]: src_ids,
                                feed_target_names[1]: sent_ids,
                                feed_target_names[2]: pos_ids,
                                feed_target_names[3]: input_mask},
                        fetch_list=fetch_targets)
            end_time = time.time()
            total_time += end_time - begin_time

            # parse outputs
            batch_result = outputs[0]
            for outer_index,single_example in enumerate(batch_result):
                persons = []
                _person = ""
                for _index, _label in enumerate(single_example):
                    if _label == 0 or _label == 1:
                        _person = _person + index_char[src_ids[outer_index][_index][0]]
                    else:
                        if _person != "" :
                            persons.append(_person)
                            _person = ""
                # print(persons)
                bgrjh = []
                record = records[(index-1)*self.args.batch_size+outer_index]
                _bgrjh = record["被告人集合"]
                for _bgr in _bgrjh:
                    if record["句子"].find(_bgr)>-1:
                        bgrjh.append(_bgr)
                if len(bgrjh)==0:
                    for _bgr in _bgrjh:
                        bgrjh.append(_bgr)

                personList = []
                for _person in persons:
                    find = False
                    for _bgr in bgrjh:
                        if _person == _bgr:
                            personList.append(_person)
                            find = True
                            break
                    if not find:
                        for _bgr in bgrjh:
                            if _bgr.find(_person)>-1:
                                personList.append(_bgr)
                                break
                            elif _person.find(_bgr)>-1:
                                personList.append(_bgr)
                                break
                personList = list(set(personList))
                #如果模型没有取到，用句子中的人名
                if len(personList) ==0:
                    for _bgr in bgrjh:
                        personList.append(_bgr)

                res.append(personList)
        return res

    def load_vocab(self):
        index_2_char = {}
        with open(args.vocab_path,"r",encoding="utf-8") as input_vocab:
            lines = input_vocab.readlines()
            for index,line in enumerate(lines):
                index_2_char[index] = line.split("\t")[0]
        return index_2_char



    def array2tensor(self,ndarray):
        """ convert numpy array to PaddleTensor"""
        assert isinstance(ndarray, np.ndarray), "input type must be np.ndarray"
        tensor = PaddleTensor(data=ndarray)
        return tensor


