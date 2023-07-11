# LSVT Dataset

## Data Downloading


The LSVT dataset [Official Website](https://rrc.cvc.uab.es/?ch=16) |
 [Download Link](https://rrc.cvc.uab.es/?ch=16&com=downloads)

> Note: Please register an account to download this dataset.

The images are split into two zipped files `train_full_images_0.tar.gz` and `train_full_images_1.tar.gz`. Both are to be downloaded. After downloading the images as above, unzip them and collect them into a single folder e.g. `train_images`.

The LSVT annotations (in JSON format) can be downloaded from [this download link](https://rrc.cvc.uab.es/?ch=16&com=downloads).
The file `train_full_labels.json` needs to be downloaded.

After downloading the images and annotations as above, the directory structure should be like as follows (ignoring the archive files):
```txt
LSVT
  |--- train_images
  |    |--- gt_0.jpg
  |    |--- gt_1.jpg
  |    |--- ...
  |--- train_full_labels.json
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name lsvt --task det \
    --image_dir path/to/LSVT/train_images/ \
    --label_dir path/to/LSVT/train_full_labels.json \
    --output_path path/to/LSVT/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `LSVT/`.

[Back to dataset converters](converters.md)
