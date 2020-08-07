from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import six
import logging
import multiprocessing
from io import open

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

exe = fluid.Executor(fluid.CPUPlace())
param_path = "/home/aistudio/predictor/checkpoints/step_5000"
save_path = "/home/aistudio/predictor/checkpoints/best_model"
prog = fluid.default_main_program()
fluid.io.load_persistables(executor=exe, dirname=param_path,
                           main_program=prog)
input_vecs = []
fetch_var = fluid.global_scope().find_var("input_ids")
fluid.io.save_inference_model(dirname=save_path, feeded_var_names=['inputs'], target_vars=[fetch_var], executor=exe)