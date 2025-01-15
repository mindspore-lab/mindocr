English| [中文](README_CN.md)

# LayoutXLM
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)

## Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->
****
LayoutXLM is the multilingual version of LayoutLMv2[<a href="#References">2</a>]. Unlike the original LayoutLM, which integrates image embeddings during the fine-tuning stage, LayoutXLM integrates visual information during the pre-training stage and utilizes a Transformer architecture to learn cross-modal interactions between text and images. Additionally, inspired by 1-D relative positional representation, the paper proposes a spatial-aware self-attention mechanism, which provides 2-D relative positional representation for token pairs. Unlike using absolute 2-D position embeddings to model document layout, relative positional embeddings can provide a larger receptive field for modeling contextual spatial relationships clearly.

As shown in the architecture diagram [Figure 1](#-Multi-modal-Encoder-with-Spatial-Aware-Self-Attention-Mechanism), LayoutXLM (LayoutLMv2) adopts a multimodal Transformer architecture as its backbone. The backbone takes text, image, and layout information as input, establishing deep cross-modal interactions. At the same time, it introduces the spatial-aware self-attention mechanism, allowing the model to better model document layout.

### Text Embedding
Tokenizing the OCR text sequence with WordPiece, each token is marked as {[A], [B]}. Then, [CLS] is added to the beginning of the sequence, and [SEP] is added to the end of each text segment. Additional [PAD] tokens are added to the end of the sequence to match the maximum sequence length, denoted as L. The final text embedding is the sum of three embeddings: token embedding representing the token itself, 1-D position embedding representing the token index, and segment embedding used to distinguish different text segments.

### Visual Embedding
Although all the required information is present in the page image, the model finds it challenging to capture detailed features through a single information-rich representation. Therefore, leveraging a CNN-based visual encoder outputs the page feature map, which also converts the page image into a fixed-length sequence. Using the ResNeXt-FPN architecture as the backbone, its parameters can be trained through backpropagation.

For a given page image I, it is resized to 224×224 before entering the visual backbone. The output feature map is then average-pooled to a fixed size: width W and height H. Afterwards, it is flattened into a visual embedding sequence of length W×H, and its dimension is aligned with the text embedding through a linear projection layer. Since the CNN-based visual backbone cannot acquire position information, 1-D position embedding is also added, which is shared with the text embedding. For segment embedding, all visual tokens are assigned to [C].

### Layout Embedding
The layout embedding layer is used to represent spatial layout information, which originates from the axis-aligned token bounding boxes obtained from OCR recognition, including the length, width, and coordinates of the boxes. Following the approach of LayoutLM, the coordinates are normalized and discretized, rounding them to integers between 0 and 1000. Two embedding layers are used to embed features along the x-axis and y-axis, respectively.

Given a normalized bounding box with xmin, xmax, ymin, ymax, width, and height, the layout embedding layer concatenates the six bounding box features to construct a 2-D position embedding, which is the layout embedding. Since CNN supports local transformations, image token embeddings can be mapped back to the original image one-to-one, without overlapping or missing tokens. Therefore, when calculating bounding boxes, visual tokens can be assigned to the corresponding grid. For special tokens such as [CLS], [SEP], and [PAD] in the text embedding, zero features for bounding boxes are appended.

### Multi-modal Encoder with Spatial-Aware Self-Attention Mechanism
The encoder concatenates visual embeddings and text embeddings into a unified sequence and adds them to the layout embeddings to blend spatial information. Following the Transformer architecture, the model constructs a multimodal encoder with a stack of multi-head self-attention layers followed by feed-forward networks. However, the original self-attention mechanism only captures absolute positional relationships between input tokens. To effectively model local invariance in document layout, it is necessary to explicitly insert relative positional information. Therefore, we propose the spatial-aware self-attention mechanism and incorporate it into the self-attention layer.

After obtaining αij from the original self-attention layer, considering the large range of positions, we model semantic relative positions and spatial relative positions as bias terms to avoid introducing too many parameters. We use three biases to represent learnable 1-D and 2-D (x, y) relative positional biases. These biases are different for each attention head but consistent across layers. Assuming a bounding box (xi, yi), the three biases are added to αij to obtain the self-attention map, and finally, the final attention scores are computed in the manner of Transformer.
 [<a href="#References">1</a>] [<a href="#References">2</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src=layoutlmv2_arch.png width=1000 />
</p>
<p align="center">
  <em> Figure 1. LayoutXLM(LayoutLMv2) architecture [<a href="#References">1</a>] </em>
</p>

## Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### Accuracy

| mindspore |  ascend driver  |   firmware   | cann toolkit/kernel |
|:---------:|:---------------:|:------------:|:-------------------:|
|   2.3.1   |    24.1.RC2     | 7.3.0.1.231  |    8.0.RC2.beta1    |

According to our experiments, the performance and accuracy evaluation（[Model Evaluation](#33-Model-Evaluation)） results of training ([Model Training](#32-Model-Training)) on the XFUND Chinese dataset are as follows:


Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode
<div align="center">

| **model name** | **cards** | **batch size** | **img/s** | **hmean** | **config**  | **weight**                                                                            |
|----------------|-----------|----------------|-----------|-----------|-----------------------------------------------------|------------------------------------------------|
| LayoutXLM    | 1         | 8              | 73.26     | 90.34%    | [yaml](../layoutxlm/ser_layoutxlm_xfund_zh.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/layoutxlm/ser_layoutxlm_base-a4ea148e.ckpt)  |
| VI-LayoutXLM    | 1         | 8              | 110.6     | 93.31%    | [yaml](../layoutxlm/ser_layoutxlm_xfund_zh.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindocr/layoutxlm/ser_layoutxlm_base-a4ea148e.ckpt)  |
</div>



## Quick Start
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### Dataset Download

[The XFUND dataset](https://github.com/doc-analysis/XFUND) is used as the experimental dataset. The XFUND dataset is a multilingual dataset proposed by Microsoft for the Knowledge-Intensive Extraction (KIE) task. It consists of seven datasets, each containing 149 training samples and 50 validation samples.

Respectively: ZH (Chinese), JA (Japanese), ES (Spanish), FR (French), IT (Italian), DE (German), PT (Portuguese)

a preprocessed [Chinese dataset](https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar) that can be directly used is provided for everyone to download.

```bash
mkdir train_data
cd train_data
wget https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar && tar -xf XFUND.tar
cd ..
```

#### Dataset Usage

After decompression, the data folder structure is as follows:

```bash
  └─ zh_train/            Training set
      ├── image/          Folder for storing images
      ├── train.json      Annotation information
  └─ zh_val/              Validation set
      ├── image/          Folder for storing images
      ├── val.json        Annotation information

```

The annotation format of this dataset is:

```bash
{
    "height": 3508,  # Image height
    "width": 2480,   # Image width
    "ocr_info": [
        {
            "text": "邮政地址:",  # Single text content
            "label": "question",  # Category of the text
            "bbox": [261, 802, 483, 859],  # Single text box
            "id": 54,  # Text index
            "linking": [[54, 60]],  # Relationships between the current text and other texts [question, answer]
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


### Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```
python tools/eval.py --config configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yaml
```


### Model Inference

To perform inference using a pre-trained model, you can utilize `tools/infer/text/predict_ser.py` for inference and visualize the results.

```
python tools/infer/text/predict_ser.py --rec_algorithm CRNN_CH --image_dir {dir of images or path of image}
```

As an example of entity recognition in Chinese forms, use the script to recognize entities in the form of `configs/kie/vi_layoutxlm/example.jpg`. The results will be stored in the `./inference_results` folder by default, and you can also customize the result storage path through the `--draw_img_save_dir` command-line parameter.

<p align="center">
  <img src="example.jpg" width=1000 />
</p>
<p align="center">
  <em> example.jpg </em>
</p>
Recognition results are as shown in the image, and the image is saved as`inference_results/example_ser.jpg`：

<p align="center">
  <img src="example_ser.jpg" width=1000 />
</p>
<p align="center">
  <em> example_ser.jpg </em>
</p>



## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. arXiv preprint arXiv:2012.14740, 2020.

[2] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei. LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding. arXiv preprint arXiv:2104.08836, 2021.
