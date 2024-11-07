[English](README.md) | 中文

# LayoutXLM
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)

## 1. 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

LayoutXLM是LayoutLMv2[<a href="#参考文献">2</a>]的多语言版本，与初版LayoutLM（图像embedding在fine-tune阶段融合）不同，LayoutXLM在预训练阶段就整合视觉信息，并利用Transformer架构学习文本和图像的跨模态交互信息。此外，受到1-D相对位置表征的启发，论文提出spatial-aware self-attention（空间感知自注意力）机制，对token pair进行2-D相对位置表征。与利用绝对2-D位置embedding建模文档布局不同的是，相对位置embedding能够清晰地为上下文空间建模提供更大的感受野。

如架构图（图 1）所示，LayoutXLM(LayoutLMv2)采用多模态Transformer架构作为backbone，backbone以文本、图像以及布局信息作为输入，建立深度跨模态交互。同时提出spatial-aware self-attention（空间感知自注意力）机制，使得模型能够更好地建模文档布局。

### Text Embedding
以WordPiece对OCR文本序列进行tokenize，并将每个token标记为{[A], [B]}。然后，将[CLS]加到序列头，[SEP]加到文本段尾。额外的[PAD]token被添加到序列尾部，使得整个序列长度与最大序列长L相同。最终text embedding是三个embedding的和，其中token embedding代表token本身，1-D position embedding表示token索引，segment embedding用于区分不同文本段。

### Visual Embedding
尽管所有需要的信息都在页面图像中，但模型很难通过单一的information-rich表征抓取其中的细节特征。因此，利用基于CNN的视觉encoder输出页面feature map，同时也能将页面图像转换为固定长度的序列。使用ResNeXt-FPN架构作为backbone，其参数可以通过反向传播训练。
对于给定的页面图像I，其被resize到224×224后进入visual backbone。之后输出的feature map通过average pooling到一个固定的尺寸：宽为W、高为H。之后将其flatten为W×H长度的visual embedding序列，并通过线性投影层将维度对齐text embedding。因为基于CNN的视觉backbone不能获取位置信息，所以还需加入1-D position embedding，这些position embedding与text embedding所共享。对于segment embedding，所有的visual token都被分配到[C]。

### Layout Embedding
Layout embedding层是用于空间布局信息表征，这种表征来自OCR识别的轴对齐token bounding box，包括box的长宽和坐标。沿用LayoutLM的方法，将坐标标准化和离散化，使其取整至0到1000，并使用两个embedding层分别embed x轴和y轴的特征。给定一个标准化后的bounding box有xmin，xmax，ymin，ymax，wildth，height，layout embedding 层concat6个bounding box 特征，构建2-Dposition embedding也就是layout embedding。CNN支持局部转换，因此图像token embedding可以一一映射回原始图像，不重叠也不遗漏。因此在计算bounding box是时，visual token可以被划分到对应的网格中。而对于text embedding中的[CLS]，[SEP]以及[PAD]特殊token，会附加全零的bounding box的feature。

### Multi-modal Encoder with Spatial-Aware Self-Attention Mechanism
Encoder concat视觉embedding和文本embedding到一个统一的序列，并与layout embedding相加以混合空间信息。遵循Transformer架构，模型用一堆多头自注意力层构建了多模态encoder，而后面则是前馈网络。但是原始的自注意力方法只会抓取输入token之间的绝对位置关系。为了有效地建模文档布局中的局部不变性，有必要显示插入相对位置位置信息。因此我们提出spatial-aware self-attention（空间感知自注意力）机制，将其加入self-attention层。在原始的self-attention层得到的αij后。考虑到位置范围较大，因此建模语义相对位置和空间相对位置，作为偏置项以免加入过多的参数。用三个偏置分别代表可学习的1-D和2-D(x, y)相对位置偏置。这些偏置在每个注意力头是不同的，但在每一层是一致的。假设boudning box(xi,yi)，算出其三个偏置项与αij相加得到自注意力map，最后按照Transformer的方式求出最终的注意力得分。
 [<a href="#参考文献">1</a>] [<a href="#参考文献">2</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src=layoutlmv2_arch.png width=1000 />
</p>
<p align="center">
  <em> 图1. LayoutXLM(LayoutLMv2)架构图 [<a href="#参考文献">1</a>] </em>
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
|  LayoutXLM   |   SER    | D910Ax1-MS2.1-G |  XFUND_zh  |  352.0 M   |      8       |             3.41             |            189.50            |          42.24           |  90.41%   | [yaml](../layoutxlm/ser_layoutxlm_xfund_zh.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/layoutxlm/ser_layoutxlm_base-a4ea148e.ckpt)  |
| VI-LayoutXLM |   SER    | D910Ax1-MS2.1-G |  XFUND_zh  |  265.7 M   |      8       |             3.06             |            169.7             |           47.2           |  93.31%   |         [yaml](ser_vi_layoutxlm_xfund_zh.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/ser_vi_layoutxlm-f3c83585.ckpt) |

</div>

### 推理端

TODO


## 3. 快速开始
### 3.1 环境及数据准备

#### 3.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 3.1.2 数据集下载
这里使用[XFUND数据集](https://github.com/doc-analysis/XFUND)做为实验数据集。 XFUN数据集是微软提出的一个用于KIE任务的多语言数据集，共包含七个数据集，每个数据集包含149张训练集和50张验证集

分别为：ZH(中文)、JA(日语)、ES(西班牙)、FR(法语)、IT(意大利)、DE(德语)、PT(葡萄牙)

这里提供了经过预处理可以直接使用的[中文数据集](https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar)供大家下载。

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
    name: layoutxlm
    pretrained: True
    num_classes: &num_classes 7
    use_visual_backbone: False
    use_float16: True
  head :
    name: TokenClassificationHead
    num_classes: 7
    use_visual_backbone: False
    use_float16: True
  pretrained:
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
* 转换PaddleOCR模型

如果要导入PaddleOCR LayoutXLM模型，可以使用`tools/param_converter.py`脚本将pdparams文件转换为mindspore支持的ckpt格式，并导入续训。

```shell
python tools/param_converter.py \
 --input_path path/to/paddleocr.pdparams \
 --json_path mindocr/models/backbones/layoutxlm/ser_vi_layoutxlm_param_map.json \
 --output_path path/to/from_paddle.ckpt
```

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数`distribute`修改为True，并运行：

```shell
# 在多个 Ascend 设备上进行分布式训练
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yaml
```


* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数`distribute`修改为False 并运行：

```shell
# CPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yaml
```

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的目录下，默认为`./tmp_kie_ser`。

### 3.3 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，然后运行：

```
python tools/eval.py --config configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yaml
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


## 4. MindSpore Lite 推理

**TODO**


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. arXiv preprint arXiv:2012.14740, 2020.

[2] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei. LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding. arXiv preprint arXiv:2104.08836, 2021.
