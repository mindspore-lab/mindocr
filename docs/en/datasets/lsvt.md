English | [中文](../../cn/datasets/lsvt_CN.md)

# LSVT Dataset - [Official Website](https://rrc.cvc.uab.es/?ch=16)

## Data Downloading

[Download Source](https://rrc.cvc.uab.es/?ch=16&com=downloads). Need to register an account to download this dataset.

<details>
    <summary>How to Download LSVT Images</summary>

The LSVT images dataset can be downloaded from [here](https://rrc.cvc.uab.es/?ch=16&com=downloads) 
The images are split into two zipped files `train_full_images_0.tar.gz` and `train_full_images_1.tar.gz`. Both are to be downloaded.

</details>

After downloading the images as above, unzip them and collect them into a single folder e.g. `train_images`.

<details>
    <summary>How to Download LSVT Annotations</summary>
    
The LSVT annotations (in JSON format) can be downloaded from [here](https://rrc.cvc.uab.es/?ch=16&com=downloads)
The file train_full_labels.json needs to be downloaded.

</details>

After downloading the images and annotations as above, the data structure should be like as follows (ignoring the archive files):
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

Then you can have an annotation file in the standard form `det_gt.txt` under the folder `LSVT/`.

[Back to README](../../../tools/dataset_converters/README.md)
