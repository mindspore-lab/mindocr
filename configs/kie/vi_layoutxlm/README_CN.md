[English](README.md) | 中文

# LayoutXLM
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)

## 模型描述
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

## 评估结果

| mindspore |  ascend driver  |   firmware   | cann toolkit/kernel |
|:---------:|:---------------:|:------------:|:-------------------:|
|   2.3.1   |    24.1.RC2     | 7.3.0.1.231  |    8.0.RC2.beta1    |

根据我们的实验，在XFUND中文数据集上训练的（[模型评估](#33-模型评估)）结果如下：

在采用图模式的ascend 910*上实验结果，mindspore版本为2.3.1
<div align="center">

| **模型名称**     | **卡数** | **单卡批量大小** | **img/s** | **hmean** | **配置**                                           | **权重**                                                                                            |
|--------------|--------|------------|-----------|-----------|--------------------------------------------------|---------------------------------------------------------------------------------------------------|
| LayoutXLM    | 1      | 8          | 73.26     | 90.34%    | [yaml](../layoutxlm/ser_layoutxlm_xfund_zh.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/layoutxlm/ser_layoutxlm_base-a4ea148e.ckpt) |
| VI-LayoutXLM | 1      | 8          | 110.6     | 93.31%    | [yaml](../layoutxlm/ser_layoutxlm_xfund_zh.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/layoutxlm/ser_layoutxlm_base-a4ea148e.ckpt) |
</div>



## 快速开始
### 环境及数据准备

#### 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 数据集下载
这里使用[XFUND数据集](https://github.com/doc-analysis/XFUND)做为实验数据集。 XFUN数据集是微软提出的一个用于KIE任务的多语言数据集，共包含七个数据集，每个数据集包含149张训练集和50张验证集

分别为：ZH(中文)、JA(日语)、ES(西班牙)、FR(法语)、IT(意大利)、DE(德语)、PT(葡萄牙)

这里提供了经过预处理可以直接使用的[中文数据集](https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar)供大家下载。

```bash
mkdir train_data
cd train_data
wget https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar && tar -xf XFUND.tar
cd ..
```

#### 数据集使用

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


### 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，然后运行：

```
python tools/eval.py --config configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yaml
```


### 模型推理

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

[1] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. arXiv preprint arXiv:2012.14740, 2020.

[2] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei. LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding. arXiv preprint arXiv:2104.08836, 2021.
