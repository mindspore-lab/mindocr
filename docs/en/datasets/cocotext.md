# COCO-Text Dataset

## Data Downloading

[Official Website](https://rrc.cvc.uab.es/?ch=5)

The COCO-Text images dataset and the annotations (in JSON format) `annotations v1.4 JSON` can be downloaded from this [link](https://rrc.cvc.uab.es/?ch=5&com=downloads).

> Note: Please register an account to download this dataset.

After downloading the images and annotations, unzip the files, after which the directory structure should be like as follows (ignoring the archive files):
```txt
COCO-Text
  |--- train_images
  |    |--- COCO_train2014_000000000036.jpg
  |    |--- COCO_train2014_000000000064.jpg
  |    |--- ...
  |--- COCO_Text.json
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name cocotext --task det \
    --image_dir path/to/COCO-Text/train_images/ \
    --label_dir path/to/COCO-Text/COCO_Text.json \
    --output_path path/to/COCO-Text/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `COCO-Text/`.

[Back to dataset converters](converters.md)
