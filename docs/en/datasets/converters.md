This document shows how to convert ocr annotation to the general format (not including LMDB) for model training.

You may also refer to [`convert_datasets.sh`](https://github.com/mindspore-lab/mindocr/blob/main/tools/convert_datasets.sh) which is a quick solution for converting annotation files of all datasets under a given directory.

<details open markdown>
<summary>To download and convert OCR datasets to the required data format, please refer to these instructions.</summary>

- [Born-Digital Images](borndigital.md)
- [CASIA-10K](casia10k.md)
- [CCPD](ccpd.md)
- [Chinese text recognition](chinese_text_recognition.md)
- [COCO-Text](cocotext.md)
- [CTW](ctw.md)
- [ICDAR2015](icdar2015.md)
- [ICDAR2019 ArT](ic19_art.md)
- [LSVT](lsvt.md)
- [MLT2017](mlt2017.md)
- [MSRA-TD500](td500.md)
- [MTWI-2018](mtwi2018.md)
- [RCTW-17](rctw17.md)
- [ReCTS](rects.md)
- [SCUT-CTW1500](ctw1500.md)
- [SROIE](sroie.md)
- [SVT](svt.md)
- [SynText150k](syntext150k.md)
- [SynthText](synthtext.md)
- [TextOCR](textocr.md)
- [Total-Text](totaltext.md)

</details>

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

### Common Dataset Format
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
        --label_dir /path/to/ic15/rec/ch4_training_word_images_gt/gt.txt \
        --output_path /path/to/ic15/rec/train/ch4_training_word_images_gt/rec_gt.txt
```

``` shell
# convert testing anotation
python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task rec \
        --label_dir /path/to/ic15/rec/ch4_test_word_images_gt/gt.txt \
        --output_path /path/to/ic15/rec/ch4_test_word_images_gt/rec_gt.txt
```

### LMDB Dataset Format

Some of the dataset can be converted to LMDB format. Currently, this is only supported for the `SynthText` and `SynthAdd` datasets.

To convert to LMDB format, please run

``` shell
python tools/dataset_converters/convert.py \
    --dataset_name synthtext \
    --task rec_lmdb \
    --image_dir /path/to/SynthText \
    --label_dir /path/to/SynthText_gt.mat \
    --output_path ST_full
```
