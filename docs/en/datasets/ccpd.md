# Chinese City Parking Dataset (CCPD) 2019

## Data Downloading

The CCPD can be downloaded from this [link](https://github.com/detectRecog/CCPD) using either the Google or BaiduYun drive links. This dataset is divided into 3 sets: train, val, test. Labels for each set can be found under the `splits` directory of the dataset.

The annotations for each image are embedded into the filename of the image. The format is described on their official website [here](https://github.com/detectRecog/CCPD#dataset-annotations).

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
    --label_dir path/to/CCPD2019/splits/train.txt \
    --output_path path/to/CCPD2019/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `CCPD2019/`.

[Back to dataset converters](converters.md)
