# Born-Digital Images Dataset


## Data Downloading

The Born-Digital dataset [Official Website](https://rrc.cvc.uab.es/?ch=1) | [Download Link](https://rrc.cvc.uab.es/?ch=1&com=downloads)

> Note: Please register an account to download this dataset.

This dataset is divided into 4 tasks: (1.1) Text Localization, (1.2) Text Segmentation, (1.3) Word Recognition, and  (1.4) End To End.  For now, we consider and download only the dataset for Task 1.1.

After downloading the images and annotations, unzip the files and rename as appropriate e.g. `train_images` for the images and `train_labels` for the ground truths, after which the directory structure should be like as follows (ignoring the archive files):
```txt
Born-Digital
  |--- train_images
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- train_labels
  |    |--- <image_name>.txt
  |    |--- <image_name>.txt
  |    |--- ...
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name borndigital --task det \
    --image_dir path/to/Born-Digital/train_images/ \
    --label_dir path/to/Born-Digital/train_labels \
    --output_path path/to/Born-Digital/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `Born-Digital/`.

[Back to dataset converters](converters.md)
