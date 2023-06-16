[English](../../en/datasets/chinese_text_recognition.md) | 中文

# 中文文字识别数据集

本文档介绍中文文本识别的数据集准备。

## 数据下载

按照 [Benchmarking-Chinese-Text-Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition) 中的设置，我们使用与 [Datasets](https://github.com/FudanVI/benchmarking-chinese-text-recognition#datasets) 中描述的相同的训练、验证和评估数据。

请下载[Download](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/README.md#download)中介绍的以下LMDB文件：

- 场景数据集：联合数据集包含 [RCTW](https://rctw.vlrlab.net/dataset), [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=downloads), [LSVT](https://rrc.cvc.uab.es/?ch=16&com=introduction), [ArT](https://rrc.cvc.uab.es/?ch=14&com=downloads), [CTW](https://link.springer.com/article/10.1007/s11390-019-1923-y)
- 网页：[MTWI](https://tianchi.aliyun.com/competition/entrance/231684/introduction)
- 文档：使用 [Text Render](https://github.com/oh-my-ocr/text_renderer) 生成
- 手写数据集：[SCUT-HCCDoc](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release)

## 数据结构整理

下载文件后，请将所有训练文件放在同一个文件夹 `training` 下，所有验证数据放在 `validation` 文件夹下，所有评估数据放在`evaluation`下。

数据结构应该是这样的：

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

## 数据集配置

要使用数据集，您可以在配置文件中指定数据集，如下所示。

### 模型训练

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

### 模型评估

```yaml
...
train:
  # 训练部分不需要修改，因不会调用
...
eval:
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/chinese-text-recognition/             # Root dir of evaluation dataset
    data_dir: evaluation/                                      # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
  ...
```

[返回](../../../tools/dataset_converters/README_CN.md)
