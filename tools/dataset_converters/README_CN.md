[English](README.md) | 中文

本文档展示了如何将OCR数据集的标注文件（不包括LMDB）转换为通用格式以进行模型训练。

您也可以参考 [`convert_datasets.sh`](../convert_datasets.sh)。这是将给定目录下所有数据集的标注文件转换为通用格式的Shell 脚本。

要下载OCR数据集，您可以参考 [ICDAR2015](../../docs/cn/datasets/icdar2015_CN.md), [MLT2017](../../docs/cn/datasets/mlt2017_CN.md), [Syntext 150k](../../docs/cn/datasets/syntext150k_CN.md), [Total Text](../../docs/cn/datasets/totaltext_CN.md)的说明。

## 文本检测/端到端文本检测

转换后的标注文件格式应为：
``` text
img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]   
``` 

以ICDAR2015（ic15）数据集为例，要将ic15数据集转换为所需的格式，请运行：

``` shell
# convert training anotation
python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task det \
        --image_dir /path/to/ic15/det/train/ch4_training_images \
        --label_dir /path/to/ic15/det/train/ch4_training_localization_transcription_gt \
        --output_path /path/to/ic15/det/train/det_gt.txt 
```

``` shell
# convert testing anotation
python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task det \
        --image_dir /path/to/ic15/det/test/ch4_test_images \
        --label_dir /path/to/ic15/det/test/ch4_test_localization_transcription_gt \
        --output_path /path/to/ic15/det/test/det_gt.txt 
```

## 文本识别
文本识别数据集的标注格式如下：

```text 
word_7.png	fusionopolis
word_8.png	fusionopolis
word_9.png	Reserve
word_10.png	CAUTION
word_11.png	citi
```
请注意，图像名称和文本标签以`\t`分隔。

要转换标注文件，请运行：
``` shell
# convert training anotation
python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task rec \
        --label_dir /path/to/ic15/rec/ch4_training_word_images_gt/gt.txt
        --output_path /path/to/ic15/rec/train/ch4_training_word_images_gt/rec_gt.txt 
```

``` shell
# convert testing anotation
python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task rec \
        --label_dir /path/to/ic15/rec/ch4_test_word_images_gt/gt.txt
        --output_path /path/to/ic15/rec/ch4_test_word_images_gt/rec_gt.txt
```