[English](../../en/tutorials/training_recognition_custom_dataset.md) | 中文

# 使用自定义数据集训练识别网络

本文档提供如何使用自定义数据集进行识别网络训练的教学，包括训练中、英文等不同语种的识别网络。

## 数据集准备

目前MindOCR识别网络支持两种输入形式，分别为
- `通用数据`：使用图像和文本文件储存的文件格式，以[RecDataset](../../../mindocr/data/rec_dataset.py)类型读取。
- `LMDB数据`: 使用[LMDB](https://www.symas.com/lmdb)储存的文件格式，以[LMDBDataset](../../../mindocr/data/rec_lmdb_dataset.py)类型读取。

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

用户可根据需求添加、删改包含在字典内的字符。值得留意的是，字符需以换行符`\n`作为分隔，并且避免相同字符出现在同一字典里。另外用户同时需要修改配置文件中的`common: num_classes`属性，确保`common: num_classes`属性为字典字符数量 + 1（在seq2seq模型中为字典字符数量 + 2)。

## 训练和评估模型

当所有数据集和配置文件准备完成，用户可开始训练使用自定义数据的模型。由于各模型训练方式不同，用户可参考对应模型介绍文档中的**模型训练**和**模型评估**章节。
