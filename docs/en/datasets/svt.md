English | [中文](../../cn/datasets/svt_CN.md)

# The Street View Text Dataset (SVT)

## Data Downloading
The Street View Text Dataset (SVT) [official website](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)

[download dataset](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)

Please download the data from the website above and unzip the file.
After unzipping the file, the data structure should be like:

```txt
svt1
 ├── img
 │   ├── 00_00.jpg 
 │   ├── 00_01.jpg
 │   ├── 00_02.jpg
 │   ├── 00_03.jpg
 │   ├── ...
 ├── test.xml
 └── train.xml
```

## Data Preparation

### For Recognition task

To prepare the data for text recognition, you can run the following command:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name  svt --task rec \
    --image_dir path/to/svt1/ \
    --label_dir path/to/svt1/train.xml \
    --output_path path/to/svt1/rec_train_gt.txt 
```

Then you can have a folder `cropped_images/` and an annotation file `rec_train_gt.txt` under the folder `svt1/`.
