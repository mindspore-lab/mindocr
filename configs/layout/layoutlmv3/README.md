English | [中文](README_CN.md)

# LayoutLMv3
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)

> [Original Repo](https://github.com/microsoft/unilm/tree/master/layoutlmv3)

## 1. Introduction
Unlike previous LayoutLM series models, LayoutLMv3 does not rely on complex CNN or Faster R-CNN networks to represent images in its model architecture. Instead, it directly utilizes image blocks of document images, thereby greatly reducing parameters and avoiding complex document preprocessing such as manual annotation of target region boxes and document object detection. Its simple unified architecture and training objectives make LayoutLMv3 a versatile pretraining model suitable for both text-centric and image-centric document AI tasks.

The experimental results demonstrate that LayoutLMv3 achieves better performance with fewer parameters on the following datasets:

- Text-centric datasets: Form Understanding FUNSD dataset, Receipt Understanding CORD dataset, and Document Visual Question Answering DocVQA dataset.
- Image-centric datasets: Document Image Classification RVL-CDIP dataset and Document Layout Analysis PubLayNet dataset.

LayoutLMv3 also employs a text-image multimodal Transformer architecture to learn cross-modal representations. Text vectors are obtained by adding word vectors, one-dimensional positional vectors, and two-dimensional positional vectors of words. Text from document images and their corresponding two-dimensional positional information (layout information) are extracted using optical character recognition (OCR) tools. As adjacent words in text often convey similar semantics, LayoutLMv3 shares the two-dimensional positional vectors of adjacent words, while each word in LayoutLM and LayoutLMv2 has different two-dimensional positional vectors.

The representation of image vectors typically relies on CNN-extracted feature grid features or Faster R-CNN-extracted region features, which increase computational costs or depend on region annotations. Therefore, the authors obtain image features by linearly mapping image blocks, a representation method initially proposed in ViT, which incurs minimal computational cost and does not rely on region annotations, effectively addressing the aforementioned issues. Specifically, the image is first resized to a uniform size (e.g., 224x224), then divided into fixed-size blocks (e.g., 16x16), and image features are obtained through linear mapping to form an image feature sequence, followed by addition of a learnable one-dimensional positional vector to obtain the image vector.[[1](#references)]

<p align="center">
  <img src=../../kie/layoutlmv3/layoutlmv3_arch.jpg width=1000 />
</p>
<p align="center">
  <em> Figure 1. LayoutLMv3 architecture [<a href="#references">1</a>] </em>
</p>

## 2. Quick Start

### 2.1 Preparation

| mindspore  |  ascend driver  |   firmware   | cann toolkit/kernel  |
|:----------:|:---------------:|:------------:|:--------------------:|
|   2.3.1    |    24.1.RC2     | 7.3.0.1.231  |    8.0.RC2.beta1     |

#### 2.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 2.1.2 PubLayNet Dataset Preparation

PubLayNet is a dataset for document layout analysis. It contains images of research papers and articles and annotations for various elements in a page such as "text", "list", "figure" etc in these research paper images. The dataset was obtained by automatically matching the XML representations and the content of over 1 million PDF articles that are publicly available on PubMed Central.

The training and validation datasets for PubLayNet can be downloaded [here](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz)

```bash
python tools/dataset_converters/convert.py \
    --dataset_name publaynet \
    --image_dir publaynet/ \
    --output_path publaynet/
```

Once the download is complete, the data can be converted to a data type in layoutlmv3 input format using the script provided by MindOCR above.

### 2.2 Model Conversion

Note: Please install torch before starting the conversion script
```bash
pip install torch
```

Download the [layoutlmv3-base-finetuned-publaynet](https://huggingface.co/HYPJUDY/layoutlmv3-base-finetuned-publaynet)  model to /path/to/layoutlmv3-base-finetuned-publaynet, and run:

```bash
python tools/param_converter_from_torch.py \
    --input_path /path/to/layoutlmv3-base-finetuned-publaynet/model_final.pt \
    --json_path configs/layout/layoutlmv3/layoutlmv3_publaynet_param_map.json \
    --output_path /path/to/layoutlmv3-base-finetuned-publaynet/from_torch.ckpt
```

### 2.3 Model Evaluation
The evaluation results on the public benchmark dataset (PublayNet) are as follows:

Experiments are tested on ascend 910* with mindspore 2.3.1 pynative mode
<div align="center">

| **model name** | **cards** | **batch size** | **img/s** | **map** | **config**                                                                                                     |
|----------------|-----------|----------------|-----------|---------|----------------------------------------------------------------------------------------------------------------|
| LayoutLMv3     | 1         | 1              | 2.7       | 94.3%   | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/layout/layoutlmv3/layoutlmv3_publaynet.yaml) |
</div>

### 2.4 Model Inference

```bash
python tools/infer/text/predict_layout.py  \
    --mode 1 \
    --image_dir {path_to_img} \
    --layout_algorithm LAYOUTLMV3 \
    --config {config_path}
```
By default, model inference results are saved in the inference_results folder

layout_res.png （Model inference visualization results）

layout_results.txt  （Model inference text results）

### 2.5 Model Training

coming soon

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. arXiv preprint arXiv:2204.08387, 2022.
