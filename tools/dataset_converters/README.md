English | [中文](README_CN.md)

This document shows how to convert ocr annotation to the general format (not including LMDB) for model training.

You may also refer to [`convert_datasets.sh`](../convert_datasets.sh) which is a quick solution for converting annotation files of all datasets under a given directory.

To download and convert OCR datasets to the required data format, please refer to the following instructions: [Chinese text recognition](../../docs/en/datasets/chinese_text_recognition.md), [CTW1500](../../docs/en/datasets/ctw1500.md), [ICDAR2015](../../docs/en/datasets/icdar2015.md), [MLT2017](../../docs/en/datasets/mlt2017.md), [SVT](../../docs/en/datasets/svt.md), [Syntext 150k](../../docs/en/datasets/syntext150k.md), [TD500](../../docs/en/datasets/td500.md), [Total Text](../../docs/en/datasets/totaltext.md), [SynthText](../../docs/en/datasets/synthtext.md).

## Text Detection/Spotting Annotation

The format of the converted annotation file should follow:
``` text
img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```

Taking ICDAR2015 (ic15) dataset as an example, to convert the ic15 dataset to the required format, please run

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


## Text Recognition Annotation

The annotation format for text recognition dataset follows
```text
word_7.png	fusionopolis
word_8.png	fusionopolis
word_9.png	Reserve
word_10.png	CAUTION
word_11.png	citi
```
Note that image name and text label are seperated by \t.

To convert, please run:
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
