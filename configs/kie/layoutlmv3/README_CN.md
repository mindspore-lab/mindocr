[English](README.md) | 中文

# LayoutLMv3
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

不同于以往的LayoutLM系列模型，在模型架构设计上，LayoutLMv3 不依赖复杂的 CNN 或 Faster R-CNN 网络来表征图像，而是直接利用文档图像的图像块，从而大大节省了参数并避免了复杂的文档预处理（如人工标注目标区域框和文档目标检测）。简单的统一架构和训练目标使 LayoutLMv3 成为通用的预训练模型，可适用于以文本为中心和以图像为中心的文档 AI 任务。

实验结果表明，LayoutLMv3在以下数据集以更少的参数量达到了更优的性能：
- 以文本为中心的数据集：表单理解FUNSD数据集、票据理解CORD数据集以及文档视觉问答DocVQA数据集。
- 以图像为中心的数据集：文档图像分类RVL-CDIP数据集以及文档布局分析PubLayNet数据集。

LayoutLMv3 还应用了文本——图像多模态 Transformer 架构来学习跨模态表征。文本向量由词向量、词的一维位置向量和二维位置向量相加得到。文档图像的文本和其相应的二维位置信息（布局信息）则利用光学字符识别（OCR）工具抽取。因为文本的邻接词通常表达了相似的语义，LayoutLMv3 共享了邻接词的二维位置向量，而 LayoutLM 和 LayoutLMv2 的每个词则用了不同的二维位置向量。

图像向量的表示通常依赖于 CNN 抽取特征图网格特征或 Faster R-CNN 提取区域特征，这些方式增加了计算开销或依赖于区域标注。因此，作者将图像块经过线性映射获得图像特征，这种图像表示方式最早在 ViT 中被提出，计算开销极小且不依赖于区域标注，有效解决了以上问题。具体来说，首先将图像缩放为统一的大小（例如224x224），然后将图像切分成固定大小的块（例如16x16），并通过线性映射获得图像特征序列，再加上可学习的一维位置向量后得到图像向量。[<a href="#参考文献">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src=layoutlmv3_arch.jpg width=1000 />
</p>
<p align="center">
  <em> 图1. LayoutLMv3架构图 [<a href="#参考文献">1</a>] </em>
</p>

## 2. 评估结果
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### 训练端

根据我们的实验，在XFUND中文数据集上训练（[模型训练](#32-模型训练)）性能和精度评估（[模型评估](#33-模型评估)）结果如下：

<div align="center">

|   **模型**   | **任务** |  **环境配置**   | **训练集** | **参数量** | **单卡批量** | **图模式单卡训练 (s/epoch)** | **图模式单卡训练 (ms/step)** | **图模式单卡训练 (FPS)** | **hmean** |                      **配置文件**                      |                                          **模型权重下载**                                          |
| :----------: | :------: | :-------------: | :--------: | :--------: | :----------: | :--------------------------: | :--------------------------: | :----------------------: | :-------: | :----------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
|  LayoutLMv3   |   SER    | D910x1-MS2.1-G |  XFUND_zh  |  265.8 M   |      8       |             19.53            |            1094.86         |          7.37           |  91.88%   | [yaml](../layoutlmv3/ser_layoutlmv3_xfund_zh.yaml) | ckpt(TODO)  |

</div>


## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集下载
这里使用[XFUND数据集](https://github.com/doc-analysis/XFUND)做为实验数据集。 XFUN数据集是微软提出的一个用于KIE任务的多语言数据集，共包含七个数据集，每个数据集包含149张训练集和50张验证集

分别为：ZH(中文)、JA(日语)、ES(西班牙)、FR(法语)、IT(意大利)、DE(德语)、PT(葡萄牙)

这里提供了已经过预处理，可以直接用于训练的[中文数据集](https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar)下载。

```bash
mkdir train_data
cd train_data
wget https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar && tar -xf XFUND.tar
cd ..
```

#### 3.1.3 数据集使用

解压文件后，数据文件夹结构如下：

```bash
  └─ zh_train/            训练集
      ├── image/          图片存放文件夹
      ├── train.json      标注信息
  └─ zh_val/              验证集
      ├── image/          图片存放文件夹
      ├── val.json        标注信息

```

该数据集的标注格式为

```bash
{
    "height": 3508, # 图像高度
    "width": 2480,  # 图像宽度
    "ocr_info": [
        {
            "text": "邮政地址:",  # 单个文本内容
            "label": "question", # 文本所属类别
            "bbox": [261, 802, 483, 859], # 单个文本框
            "id": 54,  # 文本索引
            "linking": [[54, 60]], # 当前文本和其他文本的关系 [question, answer]
            "words": []
        },
        {
            "text": "湖南省怀化市市辖区",
            "label": "answer",
            "bbox": [487, 810, 862, 859],
            "id": 60,
            "linking": [[54, 60]],
            "words": []
        }
    ]
}
```

**模型训练的数据配置**

如欲重现模型的训练，建议修改配置yaml的数据集相关字段如下：

```yaml
...
train:
  ...
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # 训练数据集根目录
    data_dir: XFUND/zh_train/image/                                     # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    label_file: XFUND/zh_train/train.json                               # 训练数据集的标签文件路径，将与`dataset_root`拼接形成完整的训练数据的标签文件路径。
...
eval:
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # 验证数据集根目录
    data_dir: XFUND/zh_val/image/                                       # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
    label_file: XFUND/zh_val/val.json                                   # 验证数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。
  ...
```

#### 3.1.4 检查配置文件
除了数据集的设置，请同时重点关注以下配置项：`system.distribute`, `system.val_while_train`, `train.loader.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_root`, `train.dataset.data_dir`, `train.dataset.label_file`,
`eval.ckpt_load_path`, `eval.dataset.dataset_root`, `eval.dataset.data_dir`, `eval.dataset.label_file`, `eval.loader.batch_size`。说明如下：

```yaml
system:
  mode:
  distribute: False                                                     # 分布式训练为True，单卡训练为False
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                                 # 边训练边验证
  drop_overflow_update: False
model:
  type: kie
  transform: null
  backbone:
    name: layoutlmv3
    pretrained: False
    checkpoints: path/to/layoutlmv3.ckpt                          # 导入ckpt位置
    num_classes: &num_classes 7
    mode: vi
...
train:
  ckpt_save_dir: './tmp_kie_ser'                                        # 训练结果（包括checkpoint、每个epoch的性能和曲线图）保存目录
  dataset_sink_mode: False
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # 训练数据集根目录
    data_dir: XFUND/zh_train/image/                                     # 训练数据集目录，将与`dataset_root`拼接形成完整训练数据集目录
    label_file: XFUND/zh_train/train.json                               # 训练数据集的标签文件路径，将与`dataset_root`拼接形成完整的训练数据的标签文件路径。
...
eval:
  ckpt_load_path: './tmp_kie_ser/best.ckpt'                             # checkpoint文件路径
  dataset_sink_mode: False
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # 验证数据集根目录
    data_dir: XFUND/zh_val/image/                                       # 验证数据集目录，将与`dataset_root`拼接形成完整验证数据集目录
    label_file: XFUND/zh_val/val.json                                   # 验证数据集的标签文件路径，将与`dataset_root`拼接形成完整的验证或评估数据的标签文件路径。
  ...
...
```

**注意:**
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当NPU卡数发生变化时，调整`batch_size`以保持全局批大小不变，或根据新的全局批大小线性调整学习率。


### 3.2 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->
* 多卡数据并行训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/kie/layoutlmv3/ser_layoutlmv3_xfund_zh.yaml
```


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/kie/layoutlmv3/ser_layoutlmv3_xfund_zh.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_kie_ser`。

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，然后运行：

```
python tools/eval.py --config configs/kie/layoutlmv3/ser_layoutlmv3_xfund_zh.yaml
```

### 3.4 模型推理

若要使用已训练的模型进行推理，可使用`tools/infer/text/predict_ser.py`进行推理并将结果进行可视化展示。

```
python tools/infer/text/predict_ser.py --rec_algorithm CRNN_CH --image_dir {dir of images or path of image}
```

以中文表单的实体识别为例，使用脚本识别`configs/kie/vi_layoutxlm/example.jpg`表单中的实体，结果将默认存放在`./inference_results`文件夹内，也可以通过`--draw_img_save_dir`命令行参数自定义结果存储路径。

<p align="center">
  <img src="example.jpg" width=1000 />
</p>
<p align="center">
  <em> example.jpg </em>
</p>
识别结果如图，图片保存为`inference_results/example_ser.jpg`：

<p align="center">
  <img src="example_ser.jpg" width=1000 />
</p>
<p align="center">
  <em> example_ser.jpg </em>
</p>



## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. arXiv preprint arXiv:2204.08387, 2022.
