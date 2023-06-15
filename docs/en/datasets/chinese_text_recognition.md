English | [中文](../../cn/datasets/chinese_text_recognition_CN.md)

# Chinese-Text-Recognition

This document introduce the dataset preparation for Chinese Text Recognition.

## Data Downloading

Following the setup in [Benchmarking-Chinese-Text-Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition), we use the same training, validation and evaliation data as described in [Datasets](https://github.com/FudanVI/benchmarking-chinese-text-recognition#datasets).

Please download the following LMDB files as introduced in [Downloads](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/README.md#download):

- scene datasets: The union dataset contains [RCTW](https://rctw.vlrlab.net/dataset), [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=downloads), [LSVT](https://rrc.cvc.uab.es/?ch=16&com=introduction), [ArT](https://rrc.cvc.uab.es/?ch=14&com=downloads), [CTW](https://link.springer.com/article/10.1007/s11390-019-1923-y)
- web: [MTWI](https://tianchi.aliyun.com/competition/entrance/231684/introduction)
- document: generated with [Text Render](https://github.com/oh-my-ocr/text_renderer)
- handwriting dataset: [SCUT-HCCDoc](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release)

## Data Structure

After downloading the files, please put all training files under the same folder `training`, all validation data under `validation` folder, and all evaluation data under `evaluation`.

The data structure should be like:

```txt
chinese-text-recognition/
├── evaluation
│   ├── document_test
|   |   ├── data.mdb
|   │   └── lock.mdb
│   ├── handwriting_test
|   |   ├── data.mdb
|   │   └── lock.mdb
│   ├── scene_test
|   |   ├── data.mdb
|   │   └── lock.mdb
│   └── web_test
|       ├── data.mdb
|       └── lock.mdb
├── training
│   ├── document_train
|   |   ├── data.mdb
|   │   └── lock.mdb
│   ├── handwriting_train
|   |   ├── data.mdb
|   │   └── lock.mdb
│   ├── scene_train
|   |   ├── data.mdb
|   │   └── lock.mdb
│   └── web_train
|       ├── data.mdb
|       └── lock.mdb
└── validation
    ├── document_val
    |   ├── data.mdb
    │   └── lock.mdb
    ├── handwriting_val
    |   ├── data.mdb
    │   └── lock.mdb
    ├── scene_val
    |   ├── data.mdb
    │   └── lock.mdb
    └── web_val
        ├── data.mdb
        └── lock.mdb

```

## Data Configuration

To use the datasets, you can specify the datasets as follow in configuration file.

### Model Training

```yaml
...
train:
  ...
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/chinese-text-recognition/                    # Root dir of training dataset
    data_dir: training/                                               # Dir of training dataset, concatenated with `dataset_root` to be the complete dir of training dataset
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/chinese-text-recognition/                    # Root dir of validation dataset
    data_dir: validation/                                             # Dir of validation dataset, concatenated with `dataset_root` to be the complete dir of validation dataset
  ...
```

### Model Evaluation

```yaml
...
train:
  # NO NEED TO CHANGE ANYTHING IN TRAIN SINCE IT IS NOT USED
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/chinese-text-recognition/             # Root dir of evaluation dataset
    data_dir: evaluation/                                      # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

[Back to README](../../../tools/dataset_converters/README.md)
