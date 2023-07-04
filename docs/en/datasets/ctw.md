# CTW Dataset
[Official Website](https://ctwdataset.github.io/)

## Data Downloading

Note: You need to fill in a form to download this dataset.

<details open markdown>
    <summary>How to Download CTW Images</summary>

The CTW images dataset can be downloaded from [here](https://ctwdataset.github.io/downloads.html).

The images are in 26 batches, i.e. 26 different .tar archived files of the format `images-trainval/ctw-trainval*.tar`. All 26 batches need to be downloaded.

</details>

<details open markdown>
    <summary>How to Download CTW Annotations</summary>

The CTW annotations (in JSON Lines format i.e. `.jsonl`) can be downloaded from [here](https://ctwdataset.github.io/downloads.html).
The annotations archived file is named `ctw-annotations.tar.gz`.

</details>

After downloading the zipped images, unzip the batches and collect all the images into a single folder e.g. `train_val/`.
After downloading the zipped annotations, unzip them.
Finally, the directory structure should look like this (ignoring the archive files):

```txt
CTW
  |--- train_val
  |    |--- 0000172.jpg
  |    |--- 0000174.jpg
  |    |--- ...
  |--- train.jsonl
  |--- val.jsonl
  |--- test_cls.jsonl
  |--- info.json
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ctw --task det \
    --image_dir path/to/CTW/train_val/ \
    --label_dir path/to/CTW/train.jsonl \
    --output_path path/to/CTW/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `CTW/`.

Note that the `label_dir` flag may be altered to prepare the validation data.

[Back to dataset converters](converters.md)
