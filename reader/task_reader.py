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

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple
import re
from data_utils.gen_train_test import item2example

import tokenization
from batching_mod import pad_batch_data
from batching import pad_batch_data as pbd2


log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            yield i.rstrip('\n').split(delimiter)
    return gen()


class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            headers = next(reader)
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None

        has_text_b = False
        if isinstance(example, dict):
            has_text_b = "text_b" in example.keys()
        else:
            has_text_b = "text_b" in example._fields

        if has_text_b:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))

        if self.is_inference:
            Record = namedtuple('Record',
                                ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple('Record', [
                'token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'
            ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                label_id=label_id,
                qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)

            log_prefix = "---------------------attention!!!-----------------------\n"
            assert (len(record.token_ids) == len(record.candidate_ids)) , log_prefix+"token_ids length: {:d} vs. candidate_ids length: {:d}\n".format(len(record.token_ids), len(record.candidate_ids))

            max_len = max(max_len, len(record.token_ids))
            if max_len >self.max_seq_len:
                max_len = self.max_seq_len
            if len(record.token_ids)>max_len:
                # token_ids 长度超过 self.max_seq_len 后进行裁剪
                record.token_ids = record.token_ids[:max_len-1]
            
            if len(record.candidate_ids)>max_len:
                record.candidate_ids = record.candidate_ids[:max_len-1]

		    # If set, the batch size will be the maximum number of tokens in one batch. Otherwise, it will be the maximum number of examples in one batch.
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                # 一个 batch 满了，就把它泄了
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        # 把剩下不足一个 batch 的都给返回
        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        examples = self._read_tsv(input_file)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []
        def f():
            try:
                for i in wrapper():
                    yield i
            except Exception as e:
                import traceback
                traceback.print_exc()
        return f

    # data_generator_json 读入数据的方法有些诡异...决定改为和train的时候一样的操作
    def data_generator_json(self,
                       input_json,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        examples_org = input_json
        # Example = namedtuple('Example',["要素原始值","句子"])
        examples = []
        for item in examples_org:
            example={}
            example["text_a"], example["text_b"], example["text_c"], _= item2example(item,phase='predict')
            # example["text_a"] = item["要素原始值"]
            # example["text_b"] = item["句子"]
            # example["text_c"] = item["句子"]
            examples.append(example)
        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            # 返回一个 batch 的数据
                            yield batch
                        all_dev_batches = []
        return wrapper


# 这是这次比赛需要使用的类
class SequenceLabelReaderWithPremise(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]
        batch_candidate_ids = [record.candidate_ids for record in batch_records]
        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len,
            return_input_mask=True,
            return_seq_lens=True)

        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        padded_position_ids = pad_batch_data(
            batch_position_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)
        
        padded_candidate_ids = pad_batch_data(
            batch_candidate_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        if self.is_inference:
            padded_label_ids = np.zeros(shape=(len(padded_token_ids),len(padded_token_ids[0]),1), dtype="int64")
        else:
            padded_label_ids = pad_batch_data(
                batch_label_ids,
                max_seq_len=self.max_seq_len,
                pad_idx=len(self.label_map) - 1)

        padded_task_ids = np.ones_like(
            padded_token_ids, dtype="int64") * self.task_id

        # 这里返回的 return_list 里面元素的顺序必须和 pyreader 的 feed_list 一致...
        # 这里的函数写的太细节了...
        # 再末尾加上 padded_candidate_ids , 注意需要同时在 finetune/sequence_label.py 的 create_model 函数中
        # pyreader 对象的 feed_list 参数末尾加上 candidate_ids
        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, padded_label_ids, batch_seq_lens, padded_candidate_ids
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        if len(tokens)!=len(labels):
            print("Pre",tokens,labels)
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            if token != "[PAD]" and token != "[CLS]" and token != "[SEP]":
                sub_token = tokenizer.tokenize(token)
            else:
                sub_token = [token]
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))
            elif label.startswith("S-"):
                b_laebl = "B-" + label[2:]
                e_label = "E-" + label[2:]
                i_label = "I-" + label[2:]
                ret_labels.extend([b_laebl] + [i_label] * (len(sub_token) - 2) + [e_label])
            elif label.startswith("E-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([i_label] * (len(sub_token) - 1) + [label])
        if len(tokens)!=len(labels):
            print("After",tokens,labels)
        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def check_zero(self, position_ids, tokens_b, tokens_a):
        # 从tokens_b 中找到 tokens_a 的 最长公共子串， 公共子串的起点作为零点
        # 利用这个零点对 position_ids 进行 校零
        find_zero = -1
        max_len = 1
        max_end = 0
        dp = np.zeros((1+len(tokens_b), 1+len(tokens_a)))
        for i in range(1,1+len(tokens_b)):
            for j in range(1, 1+len(tokens_a)):
                if tokens_b[i-1] == tokens_a[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_len:
                        max_len = int(dp[i][j])
                        max_end = i-1
        # 还有长度为 0 的 text_a ...
        if max_len >= 1:
            find_zero = max_end - max_len + 1
        # print(type(find_zero))
        # print(type(max_end))
        # print(tokens_b[find_zero: 1+max_end])
        # 找到了tokens_a 的起点
        assert (find_zero != -1 and max_len>=1), "not find tokens_a in tokens"+str(tokens_b)+'\n'+str(tokens_a)
        if find_zero != -1:
            find_zero = 2+len(tokens_a) + find_zero
            k = position_ids[find_zero]+1
            for i in range(len(position_ids)):
                if i < find_zero:
                    continue
                elif i >= find_zero and i< find_zero + max_len:
                    position_ids[i] = position_ids[find_zero]
                else:
                    position_ids[i] = k
                    k += 1
        # return position_ids


    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        if isinstance(example, tuple):
            tokens_a = tokenization.convert_to_unicode(example.text_a).split(u" ")
            tokens_b = tokenization.convert_to_unicode(example.text_b).split(u" ")
            tokens_c = tokenization.convert_to_unicode(example.text_c).split(u" ")

        # 因为 predictor.py 预测结果的时候，读入的是 txt 格式，训练的时候读入是 tsv 格式
        # 所以在数据解析的时候有一些差别
        elif isinstance(example, dict):
            tokens_a = tokenization.convert_to_unicode(example["text_a"]).split(u" ")
            tokens_b = tokenization.convert_to_unicode(example["text_b"]).split(u" ")
            tokens_c = tokenization.convert_to_unicode(example["text_c"]).split(u" ")
            
        if len(tokens_a)+len(tokens_b) +3>max_seq_length:
            # 裁剪 text_a 和 text_b 来保证最大长
            self._truncate_seq_pair(tokens_a,tokens_b,max_seq_length)
        
        if len(tokens_a)+len(tokens_c) +3>max_seq_length:
            # 裁剪 text_a 和 text_c 来保证最大长
            self._truncate_seq_pair(tokens_a,tokens_c,max_seq_length)
            
        if self.is_inference:
            labels = ["O" for i in range(len(tokens_a)+len(tokens_b))]
        else:
            labels = tokenization.convert_to_unicode(example.label).split(u" ")
        labels_a = labels[:len(tokens_a)]
        labels_b = labels[len(tokens_a):len(tokens_a)+len(tokens_b)]
        labels_c = labels[len(tokens_a):len(tokens_a)+len(tokens_b)]
        # 获得 text_a text_b 以及对应的 label ，主要是训练的时候用
        tokens_a, labels_a = self._reseg_token_label(tokens_a, labels_a, tokenizer)
        tokens_b, labels_b = self._reseg_token_label(tokens_b, labels_b, tokenizer)
        # 仿照 text_b 的操作，对 text_c 进行同理操作
        # 这个地方一开始有潜在bug，即在这个地方对 labels_b 进行了覆盖...
        # 这个地方还是有bug，就是通过 tokenizer 进行分词，会导致 [PAD] 填充被分开...
        # 这个地方果然有毒，去掉吧，全是填充物有什么好操作的...
        # 但是不操作，会出现一些奇怪的字符，所以还是看看问题在哪吧...
        # labels_c 没有用，只是做个占位符...
        tokens_c, labels_c = self._reseg_token_label(tokens_c, labels_c, tokenizer)
        
        # 必须保证 tokens_c 和 tokens_b 长度一致
        if len(tokens_c) > len(tokens_b):
            tokens_c = tokens_c[:len(tokens_b)]
        elif len(tokens_c) < len(tokens_b):
            tokens_c.extend(["[PAD]" for _ in range(len(tokens_b)-len(tokens_c))])
        else:
            pass

        while True:
            if len(tokens_a)+len(tokens_b)+3>max_seq_length:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                    labels_a.pop()
                else:
                    tokens_b.pop()
                    # 因为 tokens_c 和 tokens_b 长度相同，所以操作相同
                    tokens_c.pop()
                    labels_b.pop()

            else : break

        # tokens_a 和 tokens_b 进行拼接
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]+tokens_b+["[SEP]"]
        # 将 token 转换为 token_id ，通过 vocab 来实现
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 将 tokens_c 也进行上面一步的操作
        # 至此， candidate_ids 的长度和 tokens_ids 一致了
        tokens_c = ["[CLS]"] + ["[PAD]" for _ in tokens_a] + ["[SEP]"]+tokens_c+["[SEP]"]
        candidate_ids = tokenizer.convert_tokens_to_ids(tokens_c)

        # 将每个 token 的位置进行编码，也作为特征
        position_ids = list(range(len(token_ids)))
        # check_zero 会降低算法性能，不再使用
        # self.check_zero(position_ids, tokens_b, tokens_a)
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id] + [
            self.label_map[label] for label in labels_a
        ] + [no_entity_id]
        label_ids +=  [
            self.label_map[label] for label in labels_b
        ] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'candidate_ids', 'label_ids'])
        record = Record(
            # token_ids 是将 text_a 和 text_b 拼接在一起，然后将整体的 tokens 转换得到的 token_ids
            token_ids=token_ids,
            # text_type_ids 全部都是 0 ... 长度和 token_ids 是相同的
            text_type_ids=text_type_ids,
            # position_ids 是将 tokens 的位置进行了编码，比如 tokens=['f','x','x','k'] -> position_ids=[0,1,2,3]
            position_ids=position_ids,
            # candidate_ids 是将 tokens_c 的 tokens 转化为 对应的 token_ids
            candidate_ids=candidate_ids,
            # label_ids 是序列标注的 label 通过 data/label_map.json 映射得到的
            label_ids=label_ids)
        return record

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length-3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        if len(tokens_b)+len(tokens_a)>max_length-3:print(tokens_a,tokens_b,len(tokens_b)+len(tokens_a))

if __name__ == '__main__':
    pass
