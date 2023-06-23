English | [中文](../../cn/datasets/ccpd_CN.md)

# Chinese City Parking Dataset (CCPD) 2019
[Official Website](https://github.com/detectRecog/CCPD)

## Data Downloading

<details>
    <summary>How to Download the CCPD Dataset</summary>

The CCPD can be downloaded from [here](https://github.com/detectRecog/CCPD) using either the Google or BaiduYun drive links.

This dataset is divided into 3 train/val/test splits: the images in the `ccpd_base` folder are split into train/val sets and the images in all other folders (i.e. ccpd_blur, ccpd_challenge, ... , ccpd_weather) are exploited for the test set.

The annotations for each image are embedded into the filename of the image. The format is described on their official website [here](https://github.com/detectRecog/CCPD#dataset-annotations).

</details>

After downloading the dataset, the directory structure should be like as follows:
```txt
CCPD2019
  |--- ccpd_base
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- ccpd_blur
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- ...
  |--- ...
  |--- ...
  |--- splits
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ccpd --task det \
    --image_dir path/to/CCPD2019/ccpd_base/ \
    --output_path path/to/CCPD2019/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `CCPD2019/`.

Note that the `label_dir` flag is not required for this dataset because the labels are embedded within the images' file name.

[Back to README](../../../tools/dataset_converters/README.md)
