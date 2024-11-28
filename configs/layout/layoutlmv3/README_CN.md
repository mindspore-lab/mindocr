[English](README.md) | 中文

# LayoutLMv3
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)

> [Original Repo](https://github.com/microsoft/unilm/tree/master/layoutlmv3)

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
  <img src=../../kie/layoutlmv3/layoutlmv3_arch.jpg width=1000 />
</p>
<p align="center">
  <em> 图1. LayoutLMv3架构图 [<a href="#参考文献">1</a>] </em>
</p>


## 2. 快速开始

### 2.1 环境及数据准备

| mindspore  |  ascend driver  |   firmware   | cann toolkit/kernel  |
|:----------:|:---------------:|:------------:|:--------------------:|
|   2.3.1    |    24.1.RC2     | 7.3.0.1.231  |    8.0.RC2.beta1     |

#### 2.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 2.1.2 PubLayNet数据集准备

PubLayNet是一个用于文档布局分析的数据集。它包含研究论文和文章的图像，以及页面中各种元素的注释，如这些研究论文图像中的“文本”、“列表”、“图形”等。该数据集是通过自动匹配PubMed Central上公开的100多万篇PDF文章的XML表示和内容而获得的。

PubLayNet的训练及验证数据集可以从 [这里](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz) 下载。

```bash
python tools/dataset_converters/convert.py \
    --dataset_name publaynet \
    --image_dir publaynet/ \
    --output_path publaynet/
```

下载完成后，可以使用上述MindOCR提供的脚本将数据转换为layoutlmv3输入格式的数据类型。

### 2.2 模型转换

注：启动转换脚本前请安装torch
```bash
pip install torch
```

请下载 [layoutlmv3-base-finetuned-publaynet](https://huggingface.co/HYPJUDY/layoutlmv3-base-finetuned-publaynet) 模型到 /path/to/layoutlmv3-base-finetuned-publaynet, 然后运行:

```bash
python tools/param_converter_from_torch.py \
    --input_path /path/to/layoutlmv3-base-finetuned-publaynet/model_final.pt \
    --json_path configs/layout/layoutlmv3/layoutlmv3_publaynet_param_map.json \
    --output_path /path/to/layoutlmv3-base-finetuned-publaynet/from_torch.ckpt
```

### 2.3 模型评估
在公开基准数据集（PublayNet）上的-评估结果如下：

在采用动态图模式的ascend 910*上实验结果，mindspore版本为2.3.1
<div align="center">

| **model name** | **cards** | **batch size** | **img/s** | **map** | **config**                                                                                                     |
|----------------|-----------|----------------|-----------|---------|----------------------------------------------------------------------------------------------------------------|
| LayoutLMv3     | 1         | 1              | 2.7       | 94.3%   | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/layout/layoutlmv3/layoutlmv3_publaynet.yaml) |
</div>

### 2.4 模型推理

```bash
python tools/infer/text/predict_layout.py  \
    --mode 1 \
    --image_dir {path_to_img} \
    --layout_algorithm LAYOUTLMV3 \
    --config {config_path}
```
模型推理结果默认保存在inference_results文件夹下

layout_res.png （模型推理可视化结果）

layout_results.txt  （模型推理文本结果）

### 2.5 模型训练

coming soon

## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. arXiv preprint arXiv:2204.08387, 2022.
