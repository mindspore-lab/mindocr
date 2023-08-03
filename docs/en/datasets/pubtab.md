# PubTabNet Dataset

## Data Downloading

[Official Website](https://developer.ibm.com/exchanges/data/all/pubtabnet/)

The PubTabNet images dataset and the annotations (in JSONL format) can be downloaded from this [link](https://developer.ibm.com/exchanges/data/all/pubtabnet/).

After downloading the images and annotations, unzip the files, after which the directory structure should be like as follows (ignoring the archive files):

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

To prepare the data for Table Recognition, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name pubtab --task table \
    --image_dir path/to/pubtabnet/train/ \
    --label_dir path/to/pubtabnet/PubTabNet_2.0.0.jsonl \
    --output_path path/to/pubtabnet/pubtab_train.jsonl \
    --split train
```

```bash
python tools/dataset_converters/convert.py \
    --dataset_name pubtab --task table \
    --image_dir path/to/pubtabnet/val/ \
    --label_dir path/to/pubtabnet/PubTabNet_2.0.0.jsonl \
    --output_path path/to/pubtabnet/pubtab_val.jsonl \
    --split val
```

The generated standard annotation file `pubtab_train.jsonl` and`pubtab_val.jsonl` will now be placed under the folder `pubtabnet/`.

[Back to dataset converters](converters.md)
