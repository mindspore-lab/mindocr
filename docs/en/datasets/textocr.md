# TextOCR Dataset


## Data Downloading

The TextOCR dataset [Official Website](https://textvqa.org/textocr/) | [Download Link](https://textvqa.org/textocr/dataset/)

After downloading the images and annotations, unzip the files, after which the directory structure should be like as follows (ignoring the archive files):
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

The generated standard annotation file `det_gt.txt` will now be placed under the folder `TextOCR/`.

[Back to dataset converters](converters.md)
