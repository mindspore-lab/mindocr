# PubTabNet Dataset

## Data Downloading

PubTabNet dataset [Official Website](https://developer.ibm.com/exchanges/data/all/pubtabnet/) | [Download Link](https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz)

Download the images and annotations, unzip the files. The directory structure should be:

```txt
pubtabnet
  |--- train
  |    |--- PMC1064074_007_00.png
  |    |--- PMC1064076_003_00.png
  |    |--- ...
  |--- test
  |    |--- PMC1064127_003_00.png
  |    |--- PMC1065052_003_00.png
  |    |--- ...
  |--- val
  |    |--- PMC1064865_002_00.png
  |    |--- PMC1079806_002_00.png
  |    |--- ...
  |--- PubTabNet_2.0.0.jsonl
```

## Data Preparation

### For Table Recognition Task

To prepare the annotation for Table Recognition, run the following commands:

- Split the annotation for training set:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name pubtabnet --task table \
    --image_dir path/to/pubtabnet/train/ \
    --label_dir path/to/pubtabnet/PubTabNet_2.0.0.jsonl \
    --output_path path/to/pubtabnet/pubtab_train.jsonl \
    --split train
```

- Split the annotation for validation set:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name pubtabnet --task table \
    --image_dir path/to/pubtabnet/val/ \
    --label_dir path/to/pubtabnet/PubTabNet_2.0.0.jsonl \
    --output_path path/to/pubtabnet/pubtab_val.jsonl \
    --split val
```

- Note: the annotation for testing set is not provided

Then, the generated standard annotation file `pubtab_train.jsonl` and `pubtab_val.jsonl` will be placed under the folder `pubtabnet/`.

[Back to dataset converters](converters.md)
