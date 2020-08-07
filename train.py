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
"""Finetuning on classification tasks."""

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

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from optimization import optimization
from utils.init import init_pretraining_params, init_checkpoint
from utils.args import print_arguments, check_cuda, prepare_logger
from finetune.sequence_label import create_model, evaluate, predict, calculate_f1
from finetune_args import parser

args = parser.parse_args()
log = logging.getLogger()


def main(args):
	# 从 ernie_config_path 读取配置文件，主要是定义了网络模型的参数
	# 比如 dropout 概率、激活函数类型 etc.
	ernie_config = ErnieConfig(args.ernie_config_path)
	ernie_config.print_config()

	if args.use_cuda:
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

	# 和pytorch 里面不同，这里的 reader 并不指定是 train 还是 test 还是 dev ，而是通用的数据读取接口
	# 结合后文， paddle 里面使用的 dataloader 是 generator 格式，而不是 pytorch 中使用的 __getitem__() 的读取方式
	reader = task_reader.SequenceLabelReaderWithPremise(
		# vocab_path 是算法用到的汉字到 int 型 label 的映射
		vocab_path=args.vocab_path,
		# label_map 将算法的7种 label 到 int 型 label 的映射
		label_map_config=args.label_map_config,
		# 最大序列长度
		max_seq_len=args.max_seq_len,
		# Whether to lower case the input text. Should be True for uncased models and False for cased models.
		do_lower_case=args.do_lower_case,
		# If set, the batch size will be the maximum number of tokens in one batch. Otherwise, it will be the maximum number of examples in one batch.
		in_tokens=args.in_tokens,
		random_seed=args.random_seed,
		# task_id
		task_id=args.task_id)

	if not (args.do_train or args.do_val or args.do_test):
		raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
		                 "least one of them must be True.")

	# Program 是 Paddle Fluid 对于静态图的一种静态描述
	# 如果使用 paddle 来开发，一般会创建两个 Program
	# 一个是这里的 startup_program , 这个 Program 只运行一次来初始化参数
	# 另一个是 main_program , 这个会在每个 batch 中运行并更新权重
	startup_prog = fluid.Program()
	if args.random_seed is not None:
		startup_prog.random_seed = args.random_seed
	print(startup_prog.to_string(True))
	if args.do_train:
		# 生成 train_data_generator 来加载数据
		# train_data_generator 是 python 生成器对象
		train_data_generator = reader.data_generator(
			input_file=args.train_set,
			batch_size=args.batch_size,
			epoch=args.epoch,
			shuffle=True,
			phase="train")

		num_train_examples = reader.get_num_examples(args.train_set)

		if args.in_tokens:
			if args.batch_size < args.max_seq_len:
				raise ValueError(
					'if in_tokens=True, batch_size should greater than max_sqelen, got batch_size:%d seqlen:%d' % (
					args.batch_size, args.max_seq_len))

			max_train_steps = args.epoch * num_train_examples // (
					args.batch_size // args.max_seq_len) // dev_count
		else:
			max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

		warmup_steps = int(max_train_steps * args.warmup_proportion)
		log.info("Device count: %d" % dev_count)
		log.info("Num train examples: %d" % num_train_examples)
		log.info("Max train steps: %d" % max_train_steps)
		log.info("Num warmup steps: %d" % warmup_steps)

		train_program = fluid.Program()

		graph_vars = None


		# 该接口配合使用python的 with 语句
		# 来将 with block 里的算子和变量添加进指定的全局主程序（main program）和启动程序（startup program）
		with fluid.program_guard(train_program, startup_prog):
			# 该接口用于更改命名空间，与with语句一起使用。
			# 使用后，在with语句的上下文中使用新的命名空间
			with fluid.unique_name.guard():
				# create_model 是整体代码比较重要的一个接口，
				# 通过执行该函数可以在 train_program 中加入模型算子和变量
				# 这一点上有点类似 tensorflow
				# train_pyreader 是 DataLoader 类型，用来加载前面的 train_data_generator
				# graph_vars 包含了模型的输出，train.py 里主要用来作为 fetch_list
				# feed_target_names train.py 基本没用，在 save_model.py 和 predictor.py 里面有用
				# 还有就是 train.py 里面 exe.run() 没有指定 feed , 其实指定了的话可能感观会好一点...
				train_pyreader, graph_vars, feed_target_names = create_model(
					args,
					pyreader_name='train_reader',
					ernie_config=ernie_config)
				# 和 torch 这样的动态图框架不同，paddle 的 scheduled_lr 和 loss_scaling
				# 不需要每次迭代后进行 step() 和 backward()
				# 这些操作在创建对象的时候就已经加入 main program 里了，后面只要
				# 执行器 run() 就会同时进行 learning_rate 的调整以及 loss 的反向传播
				scheduled_lr, loss_scaling = optimization(
					loss=graph_vars["loss"],
					warmup_steps=warmup_steps,
					num_train_steps=max_train_steps,
					learning_rate=args.learning_rate,
					train_program=train_program,
					startup_prog=startup_prog,
					weight_decay=args.weight_decay,
					scheduler=args.lr_scheduler,
					use_fp16=args.use_fp16,
					use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
					init_loss_scaling=args.init_loss_scaling,
					incr_every_n_steps=args.incr_every_n_steps,
					decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
					incr_ratio=args.incr_ratio,
					decr_ratio=args.decr_ratio)
		
		# Whether to output verbose log
		if args.verbose:
			if args.in_tokens:
				# 用的多少存储？这个没找到API说明
				lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
					program=train_program,
					batch_size=args.batch_size // args.max_seq_len)
			else:
				lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
					program=train_program, batch_size=args.batch_size)
			log.info("Theoretical memory usage in training: %.3f - %.3f %s" %
			         (lower_mem, upper_mem, unit))

	if args.do_val or args.do_test:
		test_prog = fluid.Program()
		with fluid.program_guard(test_prog, startup_prog):
			# 避免和 train_program 处于同一命名空间
			with fluid.unique_name.guard():
				test_pyreader, graph_vars, feed_target_name, = create_model(
					args,
					pyreader_name='test_reader',
					ernie_config=ernie_config)
		# 这里的 clone 操作因为参数 for_test 设置成了 True ，所以会裁减掉用于训练的部分OP和变量
		test_prog = test_prog.clone(for_test=True)

	nccl2_num_trainers = 1
	nccl2_trainer_id = 0
	# 分布式，感觉V100的话应该不太会用的上
	if args.is_distributed:
		trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
		worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
		current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
		worker_endpoints = worker_endpoints_env.split(",")
		trainers_num = len(worker_endpoints)

		log.info("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}".format(worker_endpoints, trainers_num,
		                            current_endpoint, trainer_id))

		# prepare nccl2 env.
		config = fluid.DistributeTranspilerConfig()
		config.mode = "nccl2"
		t = fluid.DistributeTranspiler(config=config)
		t.transpile(
			trainer_id,
			trainers=worker_endpoints_env,
			current_endpoint=current_endpoint,
			program=train_program if args.do_train else test_prog,
			startup_program=startup_prog)
		nccl2_num_trainers = trainers_num
		nccl2_trainer_id = trainer_id

	# 创建执行器，并执行 startup_prog 来完成初始化启动程序
	exe = fluid.Executor(place)
	exe.run(startup_prog)

	# 参数初始化，需要在完成 startup_prog 后进行
	if args.do_train:
		if args.init_checkpoint and args.init_pretraining_params:
			log.info(
				"WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
				"both are set! Only arg 'init_checkpoint' is made valid.")
		# 优先加载 checkpoint ，其次才是 pretraining_param
		if args.init_checkpoint:
			init_checkpoint(
				exe,
				args.init_checkpoint,
				main_program=startup_prog,
				use_fp16=args.use_fp16)
			# for debug
			# print('for debug, program will exit here for pretraining param list echo')
			# exit(0)
		elif args.init_pretraining_params:
			init_pretraining_params(
				exe,
				args.init_pretraining_params,
				main_program=startup_prog,
				use_fp16=args.use_fp16)
	elif args.do_val or args.do_test:
		if not args.init_checkpoint:
			raise ValueError("args 'init_checkpoint' should be set if"
			                 "only doing validation or testing!")
		init_checkpoint(
			exe,
			args.init_checkpoint,
			main_program=startup_prog,
			use_fp16=args.use_fp16)

	if args.do_train:
		# 通过设置 exec_strategy ，用户可以对执行器的执行配置进行调整，比如设置执行器中线程池的大小等
		exec_strategy = fluid.ExecutionStrategy()
		if args.use_fast_executor:
			exec_strategy.use_experimental_executor = True
		# num_threads 和可用的 GPU 数 or 可用的 CPU 核心数相同
		exec_strategy.num_threads = dev_count
		# Iteration intervals to drop scope
		# 该选项表示间隔多少次迭代之后清理一次临时变量。
		# 模型运行过程中，生成的中间临时变量将被放到local execution scope中，
		# 为了避免对临时变量频繁的申请与释放，通常将其设为较大的值（比如10或者100）
		exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

		# ParallelExecutor 是 Executor 的一个升级版本，可以支持基于数据并行的多节点模型训练和测试
		train_exe = fluid.ParallelExecutor(
			use_cuda=args.use_cuda,
			loss_name=graph_vars["loss"].name,
			exec_strategy=exec_strategy,
			main_program=train_program,
			num_trainers=nccl2_num_trainers,
			trainer_id=nccl2_trainer_id)
		
		# If the data generator yields a batch each time, 
		# use DataLoader.set_batch_generator to set the data source.
		train_pyreader.set_batch_generator(train_data_generator)
	else:
		train_exe = None

	if args.do_val or args.do_test:
		test_exe = fluid.ParallelExecutor(
			use_cuda=args.use_cuda,
			main_program=test_prog,
			share_vars_from=train_exe)

	if args.do_train:
		train_pyreader.start()
		steps = 0
		graph_vars["learning_rate"] = scheduled_lr

		time_begin = time.time()
		while True:
			try:
				steps += 1
				# The steps interval to print loss
				if steps % args.skip_steps != 0:
					train_exe.run(fetch_list=[])
				else:
					fetch_list = [
						graph_vars["num_infer"].name, graph_vars["num_label"].name,
						graph_vars["num_correct"].name,
						graph_vars["loss"].name,
						graph_vars['learning_rate'].name,
					]

					out = train_exe.run(fetch_list=fetch_list)
					num_infer, num_label, num_correct, np_loss, np_lr = out
					lr = float(np_lr[0])
					loss = np_loss.mean()
					precision, recall, f1 = calculate_f1(num_label, num_infer, num_correct)

					if args.verbose:
						log.info("train pyreader queue size: %d, learning rate: %f" % (train_pyreader.queue.size(),
						                                                               lr if warmup_steps > 0 else args.learning_rate))

					current_example, current_epoch = reader.get_train_progress()
					time_end = time.time()
					used_time = time_end - time_begin
					log.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
					         "f1: %f, precision: %f, recall: %f, speed: %f steps/s"
					         % (current_epoch, current_example, num_train_examples,
					            steps, loss, f1, precision, recall,
					            args.skip_steps / used_time))
					time_begin = time.time()

				if nccl2_trainer_id == 0 and steps % args.save_steps == 0:
					save_path = os.path.join(args.checkpoints,
					                         "step_" + str(steps))
					fluid.io.save_persistables(exe, save_path, train_program)

				if nccl2_trainer_id == 0 and steps % args.validation_steps == 0:
					# evaluate dev set
					if args.do_val:
						evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
						                 current_epoch, steps)
					# evaluate test set
					if args.do_test:
						predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
						                current_epoch, steps)


			except fluid.core.EOFException:
				save_path = os.path.join(args.checkpoints, "step_" + str(steps))
				fluid.io.save_persistables(exe, save_path, train_program)
				train_pyreader.reset()
				break

	# final eval on dev set
	if nccl2_trainer_id == 0 and args.do_val:
		if not args.do_train:
			current_example, current_epoch = reader.get_train_progress()
		evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
		                 current_epoch, 'final')

	if nccl2_trainer_id == 0 and args.do_test:
		if not args.do_train:
			current_example, current_epoch = reader.get_train_progress()
		predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
		                current_epoch, 'final')


def evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                     epoch, steps):
	# evaluate dev set
	batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size
	for ds in args.dev_set.split(','):  # single card eval
		test_pyreader.set_batch_generator(
			reader.data_generator(
				ds,
				batch_size=batch_size,
				epoch=1,
				dev_count=1,
				shuffle=False))
		log.info("validation result of dataset {}:".format(ds))
		info = evaluate(exe, test_prog, test_pyreader, graph_vars,
		                args.num_labels)
		log.info(info + ', file: {}, epoch: {}, steps: {}'.format(
			ds, epoch, steps))


def predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars,
                    epoch, steps):
	test_sets = args.test_set.split(',')
	save_dirs = args.test_save.split(',')
	assert len(test_sets) == len(save_dirs), 'number of test_sets & test_save not match, got %d vs %d' % (
	len(test_sets), len(save_dirs))

	batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size
	for test_f, save_f in zip(test_sets, save_dirs):
		test_pyreader.set_batch_generator(reader.data_generator(
			test_f,
			batch_size=batch_size,
			epoch=1,
			dev_count=1,
			shuffle=False))

		save_path = save_f + '.' + str(epoch) + '.' + str(steps)
		log.info("testing {}, save to {}".format(test_f, save_path))
		res = predict(exe, test_prog, test_pyreader, graph_vars, dev_count=1)
		save_dir = os.path.dirname(save_path)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		tokenizer = reader.tokenizer
		rev_label_map = {v: k for k, v in six.iteritems(reader.label_map)}
		with open(save_path, 'w', encoding='utf8') as f:
			for s, p in res:
				# id = ' '.join(tokenizer.convert_ids_to_tokens(id))
				p = ' '.join(['%.5f' % pp[ss] for ss, pp in zip(s, p)])
				s = ' '.join([rev_label_map[ss] for ss in s])
				f.write('{}\t{}\t{}\n'.format(id, s, p))


if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
