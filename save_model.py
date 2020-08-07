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
from  json_loader import JsonLoader
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
model_g.add_arg("init_checkpoint",              str,  os.path.join(pwd_path,"checkpoints/sequecneLabelingWithPremise/step_498"),  "Init checkpoint to resume training from.")
model_g.add_arg("save_inference_model_path",    str,  os.path.join(pwd_path,"inferences"),"If set, save the inference model to this path.")
model_g.add_arg("use_fp16",                     bool, False, "Whether to resume parameters from fp16 checkpoint.")
model_g.add_arg("num_labels",                   int,  7,     "num labels for classify")
model_g.add_arg("chunk_scheme",                  str,  "IOB", "Set for sequence to chunk.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("predict_set",         str,  os.path.join(pwd_path,"data/ner_labeled_test.tsv"),  "Predict set file")
data_g.add_arg("vocab_path",          str,  os.path.join(pwd_path,"ERNIE_stable-1.0.1/vocab.txt"),  "Vocabulary path.")
data_g.add_arg("label_map_config",    str,   os.path.join(pwd_path,"data/label_map.json"),  "Label_map_config json file.")
data_g.add_arg("max_seq_len",         int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",          int,  8,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("output_prediction",    str ,   os.path.join(pwd_path,"new_data/predict_result"),"Path for store the predict_result")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",          bool,   False,  "If set, use GPU for training.")
run_type_g.add_arg("do_prediction",     bool,   True,  "Whether to do prediction on test set.")

args = parser.parse_args()
log = logging.getLogger()
# yapf: enable.
# TO SAVE THE MODEL
ernie_config = ErnieConfig(args.ernie_config_path)
ernie_config.print_config()

# below to save to infer model
predict_prog = fluid.Program()
predict_startup = fluid.Program()
with fluid.program_guard(predict_prog, predict_startup):
	with fluid.unique_name.guard():
		predict_pyreader, probs, feed_target_names, = create_model(
			args,
			pyreader_name='predict_reader',
			ernie_config=ernie_config,
			is_prediction=True)

predict_prog = predict_prog.clone(for_test=True)

if args.use_cuda:
	place = fluid.CUDAPlace(0)
	dev_count = fluid.core.get_cuda_device_count()
else:
	place = fluid.CPUPlace()
	dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(predict_startup)

if args.init_checkpoint:
	init_checkpoint(
		exe,
		args.init_checkpoint,
		main_program=predict_prog,
		use_fp16=args.use_fp16)
else:
	raise ValueError("args 'init_checkpoint' should be set for prediction!")

assert args.save_inference_model_path, "args save_inference_model_path should be set for prediction"
_, ckpt_dir = os.path.split(args.init_checkpoint.rstrip('/'))
dir_name = ckpt_dir
cur_time = time.localtime(int(time.time()))
cur_time = time.strftime("%Y-%m-%d %H:%M:%S", cur_time)
# model_path = os.path.join(args.save_inference_model_path, cur_time+dir_name)
model_path = args.save_inference_model_path
log.info("save inference model to %s" % model_path)
print([value for key, value in probs.items()])
fluid.io.save_inference_model(
	model_path,
	feed_target_names, [value for key, value in probs.items()],
	exe,
	params_filename="weights",
	main_program=predict_prog)
##end save model
