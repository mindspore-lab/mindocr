# RCTW-17 Dataset

## Data Downloading

The RCTW dataset [Official Website](https://rctw.vlrlab.net/) | [Download Link](https://rctw.vlrlab.net/dataset)

The training set is split into two zip files `train_images.zip.001` and `train_images.zip.002`. The annotations are `*_gts.zip` files.

After downloading and unzipping the images and annotations, collect the images into a single folder e.g. `train_images/`, after which the directory structure should be like as follows (ignoring the archive files):
```txt
RCTW-17
  |--- train_images
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- train_gts
  |    |--- <image_name>.txt
  |    |--- <image_name>.txt
  |    |--- ...
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name rctw17 --task det \
    --image_dir path/to/RCTW-17/train_images/ \
    --label_dir path/to/RCTW-17/train_gts \
    --output_path path/to/RCTW-17/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `RCTW-17/`.

[Back to dataset converters](converters.md)
