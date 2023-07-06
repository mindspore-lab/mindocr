# ReCTS Dataset

## Data Downloading

The ReCTS images and annotations dataset [Official Website](https://rrc.cvc.uab.es/?ch=12) | [Download Link](https://rrc.cvc.uab.es/?ch=12&com=downloads)

> Note: Please register an account to download this dataset.

After downloading the images and annotations, unzip the file, after which the directory structure should be like as follows (ignoring the archive files):
```txt
ReCTS
  |--- img
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- gt
  |    |--- <image_name>.json
  |    |--- <image_name>.json
  |    |--- ...
  |--- gt_unicode
  |    |--- <image_name>.json
  |    |--- <image_name>.json
  |    |--- ...
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name rects --task det \
    --image_dir path/to/ReCTS/img/ \
    --label_dir path/to/ReCTS/gt_unicode.json \
    --output_path path/to/ReCTS/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `ReCTS/`.

[Back to dataset converters](converters.md)
