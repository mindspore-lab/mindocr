# 使用自定义数据集训练识别网络

本文档提供如何使用自定义数据集进行识别网络训练的教学，包括训练中、英文等不同语种的识别网络。

## 数据集准备

目前MindOCR识别网络支持两种输入形式，分别为
- `通用数据`：使用图像和文本文件储存的文件格式，以[RecDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/rec_dataset.py)类型读取。
- `LMDB数据`: 使用[LMDB](https://www.symas.com/lmdb)储存的文件格式，以[LMDBDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/rec_lmdb_dataset.py)类型读取。

以下教学以使用`通用数据`文件格式为例。


### 训练集准备
请将所有训练图片置入同一文件夹，并在上层路径指定一个txt文件用来标注所有训练图片名和对应标签。txt文件例子如下

```
# 文件名	# 对应标签
word_421.png	菜肴
word_1657.png	你好
word_1814.png	cathay
```
*注意*：请将图片名和标签以 \tab 作为分隔，避免使用空格或其他分隔符。

最终训练集存放会是以下形式：

```
|-data
    |- gt_training.txt
    |- training
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

### 验证集准备
同样，请将所有验证图片置入同一文件夹，并在上层路径指定一个txt文件用来标注所有验证图片名和对应标签。最终验证集存放会是以下形式：

```
|-data
    |- gt_validation.txt
    |- validation
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

## 字典准备

为训练中、英文等不同语种的识别网络，用户需配置对应的字典。只有存在于字典中的字符会被模型正确预测。MindOCR现提供默认、中和英三种字典，其中
- `默认字典`: 只包含小写英文和数字。如用户不配置字典，该字典会被默认使用。
- `英文字典`：包括大小写英文、数字和标点符号，存放于`mindocr/utils/dict/en_dict.txt`。
- `中文字典`：包括常用中文字符、大小写英文、数字和标点符号，存放于`mindocr/utils/dict/ch_dict.txt`。

目前MindOCR暂未提供其他语种的字典配置。该功能将在新版本中推出。

## 配置文件准备

针对不同网络结构，用户需配置相对应的配置文件。现已CRNN（以Resnet34为骨干模型）为例。

### 配置英文模型

请选择`configs/rec/crnn/crnn_resnet34.yaml`做为初始配置文件，并修改当中的`train.dataset`和`eval.dataset`内容。

```yaml
...
train:
  ...
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里用通用数据方式读取
    dataset_root: dir/to/data/                                        # 数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整路径
    label_file: gt_training.txt                                       # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
...
eval:
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里用通用数据方式读取
    dataset_root: dir/to/data/                                        # 数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整路径
    label_file: gt_validation.txt                                     # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
  ...
```

并修改对应的字典位置，指向英文字典路径

```yaml
...
common:
  character_dict_path: &character_dict_path mindocr/utils/dict/en_dict.txt
...
```

由于初始配置文件的字典默认只包含小写英文和数字，为使用完整英文字典，用户需要修改对应的配置文件的`common: num_classes`属性：

```yaml
...
common:
  num_classes: &num_classes 95                                        # 数字为 字典字符数量 + 1
...
```

如网络需要输出空格，则需要修改`common.use_space_char`属性和`common: num_classes`属性如下

```yaml
...
common:
  num_classes: &num_classes 96                                        # 数字为 字典字符数量 + 空格 + 1
  use_space_char: &use_space_char True                                # 额外添加空格输出
...
```

##### 配置自定义英文字典

用户可根据需求添加、删改包含在字典内的字符。值得留意的是，字符需以换行符`\n`作为分隔，并且避免相同字符出现在同一字典里。另外用户同时需要修改配置文件中的`common: num_classes`属性，确保`common: num_classes`属性为字典字符数量 + 1（在seq2seq模型中为字典字符数量 + 2)。

### 配置中文模型

请选择`configs/rec/crnn/crnn_resnet34_ch.yaml`做为初始配置文件，同样修改当中的`train.dataset`和`eval.dataset`内容。

```yaml
...
train:
  ...
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里用通用数据方式读取
    dataset_root: dir/to/data/                                        # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整路径
    label_file: gt_training.txt                                       # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
...
eval:
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里用通用数据方式读取
    dataset_root: dir/to/data/                                        # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整路径
    label_file: gt_validation.txt                                     # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
  ...
```

并修改对应的字典位置，指向中文字典路径

```yaml
...
common:
  character_dict_path: &character_dict_path mindocr/utils/dict/ch_dict.txt
...
```

如网络需要输出空格，则需要修改`common.use_space_char`属性和`common: num_classes`属性如下

```yaml
...
common:
  num_classes: &num_classes 6625                                      # 数字为 字典字符数量 + 空格 + 1
  use_space_char: &use_space_char True                                # 额外添加空格输出
...
```

##### 配置自定义中文字典

用户可根据需求添加、删改包含在字典内的字符。值得留意的是，字符需以换行符`\n`作为分隔，并且避免相同字符出现在同一字典里。另外用户同时需要修改配置文件中的`common: num_classes`属性，确保`common: num_classes`属性为字典字符数量 + 1 (在seq2seq模型中为字典字符数量 + 2)。

## 训练模型

当所有数据集和配置文件准备完成，用户可开始训练使用自定义数据的模型。由于各模型训练方式不同，用户可参考对应模型介绍文档中的**模型训练**和**模型评估**章节。 这里仅以CRNN为例。

### 准备预训练模型

用户可以使用我们提供的预训练模型做模型做为起始训练，预训练模型往往能提升模型的收敛速度甚至精度。以中文模型为例，我们提供的预训练模型网址为<https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt>, 用户仅需在配置文件里添加`model.pretrained`添加对应网址如下

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet34
    pretrained: False
  neck:
    name: RNNEncoder
    hidden_size: 64
  head:
    name: CTCHead
    out_channels: *num_classes
  pretrained: https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt
...
```

如果遇到网络问题，用户可尝试预先把预训练模型下载到本地，把`model.pretained`改为本地地址如下

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet34
    pretrained: False
  neck:
    name: RNNEncoder
    hidden_size: 64
  head:
    name: CTCHead
    out_channels: *num_classes
  pretrained: /local_path_to_the_ckpt/crnn_resnet34_ch-7a342e3c.ckpt
...
```

如果用户不需要使用预训练模型，只需把`model.pretrained`删除即可。

### 启动训练

#### 分布式训练

在大量数据的情况下，建议用户使用分布式训练。对于在多个昇腾910设备或着GPU卡的分布式训练，请将配置参数`system.distribute`修改为True, 例如：

```shell
# 在4个 GPU/Ascend 设备上进行分布式训练
mpirun -n 4 python tools/train.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

#### 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`system.distribute`修改为False 并运行：

```shell
# CPU/GPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`train.ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

### 断点续训

如果用户期望在开始训练时同时加载模型的优化器，学习率等信息，并继续训练，可以在配置文件里面添加`model.resume`为对应的本地模型地址如下，并启动训练

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet34
    pretrained: False
  neck:
    name: RNNEncoder
    hidden_size: 64
  head:
    name: CTCHead
    out_channels: *num_classes
  resume: /local_path_to_the_ckpt/model.ckpt
...
```

### 混合精度训练

部分模型(包括CRNN, RARE, SVTR)支持混合精度训练以加快训练速度。用户可尝试把配置文件中的`system.amp_level`设为`O2`启动混合精度训练，例子如下

```yaml
system:
  mode: 0
  distribute: True
  amp_level: O2  # Mixed precision training
  amp_level_infer: O2
  seed: 42
  log_interval: 100
  val_while_train: True
  drop_overflow_update: False
  ckpt_max_keep: 5
...
```

将`system.amp_level`改为`O0`关闭混合精度训练。

## 模型评估

若要评估已训练模型的准确性，可以使用`tools/eval.py`。请在配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为`False`如下

```yaml
system:
  distribute: False # During evaluation stage, set to False
...
eval:
  ckpt_load_path: /local_path_to_the_ckpt/model.ckpt
```

然后运行：

```shell
python tools/eval.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

会得出类似模型结果如下

```log
2023-06-16 03:41:20,237:INFO:Performance: {'acc': 0.821939, 'norm_edit_distance': 0.917264}
```

其中`acc`对应的数字为模型的精确度。

## 模型推理

用户可以通过使用推理脚本快速得到模型的推理结果。请先将图片放至在同一文件夹内，然后执行

```shell
python tools/infer/text/predict_rec.py --image_dir {dir_to_your_image_data} --rec_algorithm CRNN_CH --draw_img_save_dir inference_results
```

结果会存放于`draw_img_save_dir/rec_results.txt`, 以下是部分例子

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/e220ade5-89ae-47a4-927f-2c28941a5965" width=200 />
</p>
<p align="center">
  <em> cert_id.png </em>
</p>

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/d7cfee90-d586-4796-9ebf-b56872832e71" width=400 />
</p>
<p align="center">
  <em> doc_cn3.png </em>
</p>

得出推理结果如下

```text
cert_id.png 公民身份号码44052419
doc_cn3.png 马拉松选手不会为短暂的领先感到满意，而是永远在奔跑。
```
