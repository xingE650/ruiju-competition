#!/bin/bash
export FLAGS_eager_delete_tensor_gb=0 
export FLAGS_sync_nccl_allreduce=1 
export CUDA_VISIBLE_DEVICES=0 
###训练、测试文件存放路径
export TASK_DATA_PATH=./data
###预训练模型存放路径
export MODEL_PATH=./ERNIE_stable-1.0.1
export PYTHONPATH=./ernie:${PYTHONPATH:-} 
# train
python train.py --use_cuda true --do_train true --do_val true --do_test false --batch_size 8 --init_pretraining_params ${MODEL_PATH}/params --num_labels 7 --chunk_scheme "IOB" --label_map_config ${TASK_DATA_PATH}/label_map.json --train_set ${TASK_DATA_PATH}/ner_train.tsv --dev_set ${TASK_DATA_PATH}/ner_test.tsv --test_set ${TASK_DATA_PATH}/ner_test.tsv --vocab_path ${MODEL_PATH}/vocab.txt --ernie_config_path ${MODEL_PATH}/ernie_config.json --checkpoints ./checkpoints/ --save_steps 10000 --weight_decay  0.01 --warmup_proportion 0.0 --validation_steps 100 --use_fp16 false --epoch 10 --max_seq_len 512 --learning_rate 5e-5 --skip_steps 10 --num_iteration_per_drop_scope 1 --random_seed 1

# save_inference_model
# python save_model.py --init_checkpoint ./checkpoints/sequecneLabelingWithPremise/step_7791