English | [中文](../../cn/datasets/textocr_CN.md)

# TextOCR Dataset - [Official Website](https://textvqa.org/textocr/)

## Data Downloading

[Download Source](https://textvqa.org/textocr/dataset/). Need to register an account to download this dataset.

<details>
    <summary>How to Download TextOCR Dataset</summary>

The TextOCR dataset can be downloaded from [here](https://textvqa.org/textocr/dataset/) 

</details>

After downloading the images and annotations, unzip the files, after which the data structure should be like as follows (ignoring the archive files):
```txt
TextOCR
  |--- train_val_images
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- TextOCR_0.1_train.json
  |--- TextOCR_0.1_val.json
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name textocr --task det \
    --image_dir path/to/TextOCR/train_val_images/ \
    --label_dir path/to/TextOCR/TextOCR_0.1_train.json \
    --output_path path/to/TextOCR/det_gt.txt
```

Then you can have an annotation file in the standard form `det_gt.txt` under the folder `TextOCR/`.

[Back to README](../../../tools/dataset_converters/README.md)
