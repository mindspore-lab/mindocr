# SROIE Dataset
[Official Website](https://rrc.cvc.uab.es/?ch=13)

## Data Downloading
Note: You need to register an account to download this dataset.

<details>
    <summary>How to Download SROIE Dataset</summary>

The SROIE dataset can be downloaded from [here](https://rrc.cvc.uab.es/?ch=13&com=downloads).

This dataset is divided into 3 tasks: (1) Text Localisation, (2) OCR, and (3) Key Information Extraction. For now, we consider and download the updated dataset only for Task 1.

</details>

After downloading and unzipping the dataset as above and renaming the extracted folder as appropriate e.g. `train`, the directory structure should be like as follows (ignoring the archive files):
```txt
SROIE
  |--- train
  |    |--- <image_name>.jpg
  |    |--- <image_name>.txt
  |    |--- <image_name>.jpg
  |    |--- <image_name>.txt
  |    |--- ...
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name sroie --task det \
    --image_dir path/to/SROIE/train/ \
    --label_dir path/to/SROIE/train \
    --output_path path/to/SROIE/det_gt.txt
```

The generated standard annotation file `det_gt.txt` will now be placed under the folder `SROIE/`.

[Back to dataset converters](converters.md)
