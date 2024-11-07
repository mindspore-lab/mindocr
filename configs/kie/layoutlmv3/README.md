English | [中文](README_CN.md)

# LayoutLMv3
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)


## 1. Introduction
Unlike previous LayoutLM series models, LayoutLMv3 does not rely on complex CNN or Faster R-CNN networks to represent images in its model architecture. Instead, it directly utilizes image blocks of document images, thereby greatly reducing parameters and avoiding complex document preprocessing such as manual annotation of target region boxes and document object detection. Its simple unified architecture and training objectives make LayoutLMv3 a versatile pretraining model suitable for both text-centric and image-centric document AI tasks.

The experimental results demonstrate that LayoutLMv3 achieves better performance with fewer parameters on the following datasets:

- Text-centric datasets: Form Understanding FUNSD dataset, Receipt Understanding CORD dataset, and Document Visual Question Answering DocVQA dataset.
- Image-centric datasets: Document Image Classification RVL-CDIP dataset and Document Layout Analysis PubLayNet dataset.

LayoutLMv3 also employs a text-image multimodal Transformer architecture to learn cross-modal representations. Text vectors are obtained by adding word vectors, one-dimensional positional vectors, and two-dimensional positional vectors of words. Text from document images and their corresponding two-dimensional positional information (layout information) are extracted using optical character recognition (OCR) tools. As adjacent words in text often convey similar semantics, LayoutLMv3 shares the two-dimensional positional vectors of adjacent words, while each word in LayoutLM and LayoutLMv2 has different two-dimensional positional vectors.

The representation of image vectors typically relies on CNN-extracted feature grid features or Faster R-CNN-extracted region features, which increase computational costs or depend on region annotations. Therefore, the authors obtain image features by linearly mapping image blocks, a representation method initially proposed in ViT, which incurs minimal computational cost and does not rely on region annotations, effectively addressing the aforementioned issues. Specifically, the image is first resized to a uniform size (e.g., 224x224), then divided into fixed-size blocks (e.g., 16x16), and image features are obtained through linear mapping to form an image feature sequence, followed by addition of a learnable one-dimensional positional vector to obtain the image vector.[[1](#references)]

<p align="center">
  <img src=layoutlmv3_arch.jpg width=1000 />
</p>
<p align="center">
  <em> Figure 1. LayoutLMv3 architecture [<a href="#references">1</a>] </em>
</p>

## 2. Results
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


According to our experiments, the performance and accuracy evaluation（[Model Evaluation](#33-Model-Evaluation)） results of training ([Model Training](#32-Model-Training)) on the XFUND Chinese dataset are as follows:

<div align="center">

|   **Model**   | **Task** |  **Context**   | **Dateset** | **Model Params** | **Batch size** | **Graph train 1P (s/epoch)** | **Graph train 1P (ms/step)** | **Graph train 1P (FPS)** | **hmean** |                      **Config**                      |                                          **Download**                                          |
| :----------: | :------: | :-------------: | :--------: | :--------: | :----------: | :--------------------------: | :--------------------------: | :----------------------: | :-------: | :----------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
|  LayoutLMv3   |   SER    | D910x1-MS2.1-G |  XFUND_zh  |  265.8 M   |      8       |             19.53            |            1094.86         |          7.37           |  91.88%   | [yaml](../layoutlmv3/ser_layoutlmv3_xfund_zh.yaml) | ckpt(TODO)  |

</div>



## 3. Quick Start
### 3.1 Preparation

#### 3.1.1 Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

#### 3.1.2 Dataset Download

[The XFUND dataset](https://github.com/doc-analysis/XFUND) is used as the experimental dataset. The XFUND dataset is a multilingual dataset proposed by Microsoft for the Knowledge-Intensive Extraction (KIE) task. It consists of seven datasets, each containing 149 training samples and 50 validation samples.

Respectively: ZH (Chinese), JA (Japanese), ES (Spanish), FR (French), IT (Italian), DE (German), PT (Portuguese)

a preprocessed [Chinese dataset](https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar) that can be directly used is provided for everyone to download.

```bash
mkdir train_data
cd train_data
wget https://download.mindspore.cn/toolkits/mindocr/vi-layoutxlm/XFUND.tar && tar -xf XFUND.tar
cd ..
```

#### 3.1.3 Dataset Usage

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

**The data configuration for model training.**

If you want to reproduce the training of the model, it is recommended to modify the dataset-related fields in the configuration YAML file as follows:

```yaml
...
train:
  ...
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # Root directory of the training dataset
    data_dir: XFUND/zh_train/image/                                     # Directory of the training dataset, concatenated with `dataset_root` to form the complete directory of the training dataset
    label_file: XFUND/zh_train/train.json                               # Path to the label file of the training dataset, concatenated with `dataset_root` to form the complete path of the label file of the training dataset
...
eval:
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # Root directory of the validation dataset
    data_dir: XFUND/zh_val/image/                                       # Directory of the validation dataset, concatenated with `dataset_root` to form the complete directory of the validation dataset
    label_file: XFUND/zh_val/val.json                                   # Path to the label file of the validation dataset, concatenated with `dataset_root` to form the complete path of the label file of the validation dataset
  ...

```

#### 3.1.4 Check YAML Config Files
Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`, `train.ckpt_save_dir`, `train.dataset.dataset_path`, `eval.ckpt_load_path`, `eval.dataset.dataset_path`, `eval.loader.batch_size`. Explanations of these important args:

```yaml
system:
  mode:
  distribute: False                                                     # `True` for distributed training, `False` for standalone training
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                                 # Validate while training
  drop_overflow_update: False
model:
  type: kie
  transform: null
  backbone:
    name: layoutlmv3
  head:
    name: TokenClassificationHead
    num_classes: 7
    use_visual_backbone: True
    use_float16: True
  pretrained:
...
train:
  ckpt_save_dir: './tmp_kie_ser'                                        # The training result (including checkpoints, per-epoch performance and curves) saving directory
  dataset_sink_mode: False
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # Path of training dataset
    data_dir: XFUND/zh_train/image/                                     # Path of training dataset data dir
    label_file: XFUND/zh_train/train.json                               # Path of training dataset label file
...
eval:
  ckpt_load_path: './tmp_kie_ser/best.ckpt'                             # checkpoint file path
  dataset_sink_mode: False
  dataset:
    type: KieDataset
    dataset_root: path/to/dataset/                                      # Path of evaluation dataset
    data_dir: XFUND/zh_val/image/                                       # Path of evaluation dataset data dir
    label_file: XFUND/zh_val/val.json                                   # Path of evaluation dataset label file
...
  ...
...
```

**Notes:**
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of NPUs, or adjust the learning rate linearly to a new global batch size.


### 3.2 Model Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->
* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please modify the configuration parameter `distribute` as True and run:

```shell
# distributed training on multiple Ascend devices
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/kie/layoutlmv3/ser_layoutlmv3_xfund_zh.yaml
```


* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please modify the configuration parameter`distribute` as False and run:

```shell
# standalone training on a CPU/Ascend device
python tools/train.py --config configs/kie/layoutlmv3/ser_layoutlmv3_xfund_zh.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir`. The default directory is `./tmp_kie_ser`.

### 3.3 Model Evaluation

To evaluate the accuracy of the trained model, you can use `eval.py`. Please set the checkpoint path to the arg `ckpt_load_path` in the `eval` section of yaml config file, set `distribute` to be False, and then run:

```
python tools/eval.py --config configs/kie/layoutlmv3/ser_layoutlmv3_xfund_zh.yaml
```

### 3.4 Model Inference

To perform inference using a pre-trained model, you can utilize `tools/infer/text/predict_ser.py` for inference and visualize the results.

```
python tools/infer/text/predict_ser.py --rec_algorithm CRNN_CH --image_dir {dir of images or path of image}
```

As an example of entity recognition in Chinese forms, use the script to recognize entities in the form of `configs/kie/vi_layoutxlm/example.jpg`. The results will be stored in the `./inference_results` folder by default, and you can also customize the result storage path through the `--draw_img_save_dir` command-line parameter.

<p align="center">
  <img src="../vi_layoutxlm/example.jpg" width=1000 />
</p>
<p align="center">
  <em> example.jpg </em>
</p>
Recognition results are as shown in the image, and the image is saved as`inference_results/example_ser.jpg`：

<p align="center">
  <img src="../vi_layoutxlm/example_ser.jpg" width=1000 />
</p>
<p align="center">
  <em> example_ser.jpg </em>
</p>



## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. arXiv preprint arXiv:2204.08387, 2022.
