English | [中文](../../cn/datasets/rctw17_CN.md)

# RCTW-17 Dataset - [Official Website](https://rctw.vlrlab.net/)

## Data Downloading

[Download Source](https://rctw.vlrlab.net/dataset)

<details>
    <summary>How to Download RCTW-17 Dataset</summary>

The RCTW dataset can be downloaded from [here](https://rctw.vlrlab.net/dataset) 

The training set is split into two zip files `train_images.zip.001` and `train_images.zip.002`. The annotations are `*_gts.zip` files.

</details>

After downloading and unzipping the images and annotations, collect the images into a single folder e.g. `train_images/`, after which the data structure should be like as follows (ignoring the archive files):
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

Then you can have an annotation file in the standard form `det_gt.txt` under the folder `RCTW-17/`.

[Back to README](../../../tools/dataset_converters/README.md)
