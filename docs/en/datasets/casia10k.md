English | [中文](../../cn/datasets/casia10k_CN.md)

# CASIA-10K Dataset - [Official Website](http://www.nlpr.ia.ac.cn/pal/CASIA10K.html)

## Data Downloading

<details>
    <summary>How to Download COCO-Text Images and Annotations</summary>

The COCO-Text images dataset can be downloaded from [here](http://www.nlpr.ia.ac.cn/pal/CASIA10K.html) 

</details>

After downloading the file as above, unzip it, after which the directory structure should be like as follows (ignoring the archive file):

```txt
CASIA-10K
  |--- test
  |    |--- PAL00001.jpg
  |    |--- PAL00001.txt
  |    |--- PAL00005.jpg
  |    |--- PAL00005.txt
  |    |--- ...
  |--- train
  |    |--- PAL00003.jpg
  |    |--- PAL00003.txt
  |    |--- PAL00006.jpg
  |    |--- PAL00006.txt
  |    |--- ...
  |--- CASIA-10K_test.txt
  |--- CASIA-10K_train.txt
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name casia10k --task det \
    --image_dir path/to/CASIA-10K/train/ \
    --label_dir path/to/CASIA-10K/train \
    --output_path path/to/CASIA-10K/det_gt.txt
```

Then you can have an annotation file in the standard form `det_gt.txt` under the folder `CASIA-10K/` for the train set.

[Back to README](../../../tools/dataset_converters/README.md)