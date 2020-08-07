## Project

这是参加睿聚杯NER比赛的项目，主要是copy的百度提供的baseline，然后整理一下文档，方便和同学学习和交流。

## Run

训练：

```bash
bash run.sh
```

将训练模型转换为推理模型：

```bash
python save_model.py --init_checkpoint ./checkpoints/sequecneLabelingWithPremise/step_xxxx
```

## Submit

压缩的时候不要压缩预训练模型和checkpoints（有500MB的提交限制），使用下面的命令操作：
```bash
zip -r  predictor.zip ./ -rx=ERNIE_stable-1.0.1/params/* -rx=checkpoints/* -rx=__pycache__/*
```

比赛需要提交预测代码，测试环境和这个项目环境相同，需要提供推理模型来供调用。比赛测试平台是CPU平台，而且评估结果会受到提交人数
的影响，我第一次提交差不多花了5个小时才返回结果，群里有的老哥7-8个小时才返回结果。

## Model

baseline提供的算法模型比较简单。
- 模型将“要素原始值”+“句子”作为输入，然后按照序列标注常用的方法来组织输入：词嵌入+位置嵌入+句子类型（这个没看懂有什么用）
- 采用transformer作为backbone，将transformer-encoder的输出经过一层fc+softmax即得到最终的预测结果。
- 训练loss使用的交叉熵loss。

## Folder&&File

### data_utils

实际的数据需要处理一下，比如给的是txt格式的文档，但是baseline的代码都是读取的tsv格式的文档，包括一些数据清洗的工作，主要通过
这部分代码实现。

运行：
```python
cd predictor
python data_utils/file_check.py
python data_utils/gen_train_test.py
```

#### file_check.py

从baseline中拷贝的，用来滤除缺失信息的样本。得到有问题的`error_docs.json`和没问题的`good_docs.json`。

#### gen_train_test.py

看群里讨论说的，实际上比较重要训练数据有要素原始值，句子，然后就是NER的标签了，baseline提供了一个例子，我放在`data_utils/ner_train_example.txt`
里面。所以`gen_train_test.py`主要完成从提供的json格式数据中拆分得到`ner_train.tsv`和`ner_test.tsv`两个文件。

### ERNIE_stable-1.0.1

这个文件夹包含了[ERNIE](https://github.com/PaddlePaddle/ERNIE)预训练模型，在训练的时候加载可以缩短训练时间，提高训练效果（有点像Bert）。

然后还有一些配置文件，主要定义了transformer-encoder模型的一些参数。

### data

存放训练集和验证集的位置。

### finetune

这个文件夹下用到的只有`sequence_label.py`。 `sequence_label.py`主要实现了transformer-encoder下游的序列预测任务，以及提供了`create_model`接口生成的。训练结束后，可以通过下面的指令生成预测模型文件：

### model

需要关注两个文件：
- transformer_encoder.py 这是模型的核心代码，主要实现了transformer module，以及残差连接、dropout、layer normalization；
- ernie.py 将transformer_encoder.py的一些方法封装起来；

### utils

主要关注一下：
- args.py 提供了命令行参数实现；
- init.py 提供了模型初始化代码；
- fp16.py 代码里也用到了，但是比较底层我没看；

### reader

主要关注一下：
- task_reader.py 提供了数据读取的代码；

### checkpoints

包含了测试结果文件`test_result.x.yyyy`和保存的模型（应该是包含了定期保存的模型，以及验证集上效果最好的模型）；

### batching.py && batching_mod.py

将数据封装成 batch 格式，和`task_reader.py`一起使用；

### tokenization.py

提供了一些单词级的操作；

### finetune_args.py

提供了命令行参数实现；

### json_loader.py

save_model.py 会用到；

### test.py

模拟最终评估平台的执行，可以用来测试`predictor.py`执行的正确性和效果；

### predictor.py

最终评估平台主要是调用`predictor.py`来完成代码效果的评估；

### optimization.py

封装了学习率和loss优化的定义，主要在模型训练的时候用到；

### train.py

具体的注解可以看train.py的注释。

### inferences

包含了predictor.py所需的预测模型文件:
- `__model__`
- `weights`

这两个文件是通过paddle的`save_inference_model`接口生成的。训练结束后，可以通过下面的指令生成预测模型文件：
```bash
python save_model.py --init_checkpoint ./checkpoints/sequecneLabelingWithPremise/step_xxxx
```

