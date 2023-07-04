# ICDAR2019 Dataset
[Official Website](https://rrc.cvc.uab.es/?ch=14)

## Data Downloading

Note: You need to register an account to download this dataset.

<details open markdown>
    <summary>How to Download ICDAR2019 ArT Images and Annotations</summary>

The ICDAR2019 ArT images and annotations dataset can be downloaded from [here](https://rrc.cvc.uab.es/?ch=14&com=downloads).

For the images, the archived file `train_images.tar.gz` from the section "Task 1 and Task 3" needs to be downloaded. For the annotations, the .JSON file `train_labels.json` from the same section needs to be downloaded.

</details>

After downloading the dataset, unzip the files, after which the directory structure should be like as follows (ignoring the archive files):
```txt
ICDAR2019-ArT
  |--- train_images
  |    |--- train_images
  |    |    |--- gt_0.jpg
  |    |    |--- gt_1.jpg
  |    |    |--- ...
  |--- train_labels.json
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ic19_art --task det \
    --image_dir path/to/ICDAR2019-ArT/train_images/train_images/ \
    --label_dir path/to/ICDAR2019-ArT/train_labels.json \
    --output_path path/to/ICDAR2019-ArT/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `ICDAR2019-ArT/`.

[Back to dataset converters](converters.md)
