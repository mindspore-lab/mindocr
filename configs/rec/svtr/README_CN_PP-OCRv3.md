# SVTR-PPOCRv3
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

主流的场景文字识别模型通常包含两个基本构建部分，一个视觉模型用于特征提取和一个序列模型用于文本转换。虽然这种混合架构非常准确，但也相对复杂和低效。因此，作者提出了一种新的方法：单一视觉模型。这种方法在图形标记化（image tokenization）框架下建立，完全抛弃了顺序的建模方式。作者的方法将图像划分成小的补丁，并通过逐层组件级别的混合、合并和/或组合进行操作以实现层级。作者还设计了全局和局部混合块以识别多颗粒度的字符组件模式，从而进行字符识别。作者实验了英文和中文场景文本识别任务，结果表明作者的模型SVTR是有效的。作者的大型模型SVTR-L在英文方面能提供高准确度的性能，在中文方面也表现优越且速度更快。作者的小型模型SVTR-T在推断方面也有很好的表现。[<a href="#参考文献">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://github.com/zhtmike/mindocr/assets/8342575/27da30e5-f0af-4a11-afc8-a902785e44c1" width=450 />
</p>
<p align="center">
  <em> 图1. SVTR结构 [<a href="#参考文献">1</a>] </em>
</p>

该SVTR-PPOCRv3网络参考自PP-OCRv3 [<a href="#参考文献">2</a>] 的识别模块。其中针对SVTR-Tiny的优化主要有：
 - SVTR_LCNet：轻量级文本识别网络；
 - GTC：Attention指导CTC训练策略；
 - TextConAug：挖掘文字上下文信息的数据增广策略；
 - TextRotNet：自监督的预训练模型；
 - UDML：联合互学习策略；
 - UIM：无标注数据挖掘方案。

## 2. 权重转换

如您已经有采用PaddleOCR训练好的PaddlePaddle模型，想在MindOCR下直接进行推理或进行微调续训，您可以将训练好的模型转换为MindSpore格式的ckpt文件。

运行param_converter.py脚本，输入需要进行转换的pdparams文件、权重名字对应关系json文件和ckpt输出路径，即可进行权重转换。

其中，权重名字对应关系json文件所包含的key和value分别为MindSpore参数名称和Paddle参数名称。

```shell
python tools/param_converter.py \
    -iuput_path path/to/paddleocr.pdparams \
    -json_path configs/rec/svtr/svtr_ppocrv3_ch_param_map.json \
    -output_path path/to/output.ckpt
```


## 3. 模型训练
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集准备

目前MindOCR识别网络支持两种输入形式，分别为
- `LMDB数据`: 使用[LMDB](https://www.symas.com/lmdb)储存的文件格式，以[LMDBDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/rec_lmdb_dataset.py)类型读取。
- `通用数据`：使用图像和文本文件储存的文件格式，以[RecDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/rec_dataset.py)类型读取。


**模型训练的数据配置**

如您使用的是LMDB格式的数据集，请修改配置文件对您的数据路径进行设置。

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset                                                 # 文件读取方式，这里选择用LMDB数据方式读取
    dataset_root: dir/to/data_lmdb_release/                           # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
...
eval:
  dataset:
    type: LMDBDataset                                                 # 文件读取方式，这里选择用LMDB数据方式读取
    dataset_root: dir/to/data_lmdb_release/                           # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
  ...
```

如您使用的是通用文字识别数据格式的数据集，请修改配置文件对您的数据路径进行设置。

```yaml
...
train:
  ...
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里选择用通用数据方式读取
    dataset_root: dir/to/data/                                        # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    label_file: gt_training.txt                                       # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
...
eval:
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里用通用数据方式读取
    dataset_root: dir/to/data/                                        # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
    label_file: gt_validation.txt                                     # 验证数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
...
```

以使用`通用数据`文件格式为例。

**训练集准备**

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

**验证集准备**

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


#### 3.1.3 字典准备

为训练中、英文等不同语种的识别网络，用户需配置对应的字典。只有存在于字典中的字符会被模型正确预测。MindOCR现提供默认、中和英三种字典，其中
- `默认字典`: 只包含小写英文和数字。如用户不配置字典，该字典会被默认使用。
- `英文字典`：包括大小写英文、数字和标点符号，存放于`mindocr/utils/dict/en_dict.txt`。
- `中文字典`：包括常用中文字符、大小写英文、数字和标点符号，存放于`mindocr/utils/dict/ch_dict.txt`。

目前MindOCR暂未提供其他语种的字典配置。该功能将在新版本中推出。

以使用中文字典`ch_dict.txt`为例

修改对应的字典位置，指向字典路径

```yaml
...
common:
  character_dict_path: &character_dict_path mindocr/utils/dict/ch_dict.txt
...
```


由于初始配置文件的字典默认只包含小写英文和数字，为使用完整中文字典，用户需要修改对应的配置文件的`common: num_classes`属性：

```yaml
...
common:
  num_classes: &num_classes 6623                                        # 数字为 字典字符数量
...
```

如网络需要输出空格，则需要修改`common.use_space_char`属性和`common: num_classes`属性如下

```yaml
...
common:
  num_classes: &num_classes 6624                                        # 数字为 字典字符数量 + 空格
  use_space_char: &use_space_char True                                  # 额外添加空格输出
...
```


#### 3.1.4 配置文件准备

除了数据集的设置，请同时重点关注以下变量的配置：`system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`。说明如下：

```yaml
system:
  distribute: True                                                    # 分布式训练为True，单卡训练为False
  amp_level: 'O2'
  amp_level_infer: "O2"
  seed: 42
  val_while_train: True                                               # 边训练边验证
  drop_overflow_update: True
common:
  ...
  batch_size: &batch_size 128                                         # 训练批大小
...
train:
  ckpt_save_dir: './tmp_rec'                                          # 训练结果（包括checkpoint、每个epoch的性能和曲线图）保存目录
  dataset_sink_mode: False
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里选择用通用数据方式读取
    dataset_root: dir/to/data/                                        # 训练数据集根目录
    data_dir: training/                                               # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    label_file: gt_training.txt                                       # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
...
eval:
  ckpt_load_path: './tmp_rec/best.ckpt'                               # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: RecDataset                                                  # 文件读取方式，这里用通用数据方式读取
    dataset_root: dir/to/data/                                        # 验证数据集根目录
    data_dir: validation/                                             # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
    label_file: gt_validation.txt                                     # 训练数据集标签摆放位置，将与`dataset_root`拼接形成完整路径
...
  loader:
      shuffle: False
      batch_size: 128                                                 # 验证或评估批大小
...
```


### 3.2 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

用户可以使用我们提供的预训练模型做模型做为起始训练，预训练模型往往能提升模型的收敛速度甚至精度。以中文模型为例，我们提供的预训练模型网址为<https://download-mindspore.osinfra.cn/toolkits/mindocr/svtr/svtr_lcnet_ppocrv3-6c1d0085.ckpt>, 用户仅需在配置文件里添加`model.pretrained`添加对应网址如下

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: mobilenet_v1_enhance
    scale: 0.5
    last_conv_stride: [ 1, 2 ]
    last_pool_type: avg
    last_pool_kernel_size: [ 2, 2 ]
    pretrained: False
  head:
    name: MultiHead
    out_channels_list:
      - CTCLabelDecode: 6625
      - SARLabelDecode: 6627
    head_list:
      - CTCHead:
          Neck:
            name: svtr
          out_channels: *num_classes
      - SARHead:
          enc_dim: 512
          max_text_length: *max_text_len
  pretrained: https://download-mindspore.osinfra.cn/toolkits/mindocr/svtr/svtr_lcnet_ppocrv3-6c1d0085.ckpt
...
```

如果遇到网络问题，用户可尝试预先把预训练模型下载到本地，把`model.pretained`改为本地地址如下

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: mobilenet_v1_enhance
    scale: 0.5
    last_conv_stride: [ 1, 2 ]
    last_pool_type: avg
    last_pool_kernel_size: [ 2, 2 ]
    pretrained: False
  head:
    name: MultiHead
    out_channels_list:
      - CTCLabelDecode: 6625
      - SARLabelDecode: 6627
    head_list:
      - CTCHead:
          Neck:
            name: svtr
          out_channels: *num_classes
      - SARHead:
          enc_dim: 512
          max_text_length: *max_text_len
  pretrained: path/to/svtr_lcnet_ppocrv3-6c1d0085.ckpt
...
```

如果用户不需要使用预训练模型，只需把`model.pretrained`删除即可。

* 分布式训练

在大量数据的情况下，建议用户使用分布式训练。对于在多个昇腾910设备的分布式训练，请将配置参数`system.distribute`修改为True, 例如：

```shell
# 在多个 Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/svtr/svtr_ppocrv3_ch.yaml
```


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/svtr/svtr_ppocrv3_ch.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_rec`。

* 断点续训

如果用户期望在开始训练时同时加载模型的优化器，学习率等信息，并继续训练，可以在配置文件里面添加`model.resume`为对应的本地模型地址如下，并启动训练

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: mobilenet_v1_enhance
    scale: 0.5
    last_conv_stride: [ 1, 2 ]
    last_pool_type: avg
    last_pool_kernel_size: [ 2, 2 ]
    pretrained: False
  head:
    name: MultiHead
    out_channels_list:
      - CTCLabelDecode: 6625
      - SARLabelDecode: 6627
    head_list:
      - CTCHead:
          Neck:
            name: svtr
          out_channels: *num_classes
      - SARHead:
          enc_dim: 512
          max_text_length: *max_text_len
  resume: path/to/model.ckpt
...
```

* 混合精度训练

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

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`tools/eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，设置`distribute`为`False`如下

```yaml
system:
  distribute: False # During evaluation stage, set to False
...
eval:
  ckpt_load_path: path/to/model.ckpt
```

然后运行：

```shell
python tools/eval.py --config configs/rec/svtr/svtr_ppocrv3_ch.yaml
```

### 3.4 模型推理

用户可以通过使用推理脚本快速得到模型的推理结果。请先将图片放至在--image_dir指定的同一文件夹内，然后执行

```shell
python tools/infer/text/predict_rec.py --image_dir {dir_to_your_image_data} --rec_algorithm SVTR_PPOCRv3_CH --rec_image_shape 3,48,320 --draw_img_save_dir inference_results
```

识别结果默认会存放于`./inference_results/rec_results.txt`

如您想对图片进行串联推理，即先对图片进行文字检测，再对检测出的文字进行文字识别，您可以运行如下命令

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} --det_algorithm DB_PPOCRv3 --rec_algorithm SVTR_PPOCRv3_CH --rec_image_shape 3,48,320 --draw_img_save_dir inference_results
```
检测、识别的结果默认存放于`./inference_results/system_result.txt`下，如您想对结果进行可视化，在命令中加入`--visualize_output True`即可


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang. SVTR: Scene Text Recognition with a Single Visual Model. arXiv preprint arXiv:2205.00159, 2022.

[2] PaddleOCR PP-OCRv3 https://github.com/PaddlePaddle/PaddleOCR/blob/344b7594e49f0fbb4d6578bd347505664ed728bf/doc/doc_ch/PP-OCRv3_introduction.md#2
