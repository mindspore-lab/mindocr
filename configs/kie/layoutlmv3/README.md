English | [中文](README_CN.md)

# LayoutLMv3
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)


## 1. Introduction
Unlike previous LayoutLM series models, LayoutLMv3 does not rely on complex CNN or Faster R-CNN networks to represent images in its model architecture. Instead, it directly utilizes image blocks of document images, thereby greatly reducing parameters and avoiding complex document preprocessing such as manual annotation of target region boxes and document object detection. Its simple unified architecture and training objectives make LayoutLMv3 a versatile pretraining model suitable for both text-centric and image-centric document AI tasks.

It is applicable to datasets focused on text-centric tasks such as form understanding FUNSD dataset, receipt understanding CORD dataset, document visual question answering DocVQA dataset, as well as image-centric datasets such as document image classification RVL-CDIP dataset, and document layout analysis PubLayNet dataset. Experimental results demonstrate that LayoutLMv3 achieves superior performance with fewer parameters on these datasets.

LayoutLMv3 also employs a text-image multimodal Transformer architecture to learn cross-modal representations. Text vectors are obtained by adding word vectors, one-dimensional positional vectors, and two-dimensional positional vectors of words. Text from document images and their corresponding two-dimensional positional information (layout information) are extracted using optical character recognition (OCR) tools. As adjacent words in text often convey similar semantics, LayoutLMv3 shares the two-dimensional positional vectors of adjacent words, while each word in LayoutLM and LayoutLMv2 has different two-dimensional positional vectors.

The representation of image vectors typically relies on CNN-extracted feature grid features or Faster R-CNN-extracted region features, which increase computational costs or depend on region annotations. Therefore, the authors obtain image features by linearly mapping image blocks, a representation method initially proposed in ViT, which incurs minimal computational cost and does not rely on region annotations, effectively addressing the aforementioned issues. Specifically, the image is first resized to a uniform size (e.g., 224x224), then divided into fixed-size blocks (e.g., 16x16), and image features are obtained through linear mapping to form an image feature sequence, followed by addition of a learnable one-dimensional positional vector to obtain the image vector.[[1](#references)]

<p align="center">
  <img src=layoutlmv3_arch.jpg width=1000 />
</p>
<p align="center">
  <em> 图1. LayoutLMv3 architecture [<a href="#references">1</a>] </em>
</p>