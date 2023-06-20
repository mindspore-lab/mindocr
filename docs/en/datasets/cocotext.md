English | [中文](../../cn/datasets/cocotext_CN.md)

# COCO-Text Dataset - [Official Website](https://vision.cornell.edu/se3/coco-text-2/)

## Data Downloading

[Download Source](https://rrc.cvc.uab.es/?ch=5&com=downloads). Need to register an account to download this dataset.

<details>
    <summary>How to Download COCO-Text Images</summary>

The COCO-Text images dataset can be downloaded from [here](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=5&f=aHR0cDovL21zdm9jZHMuYmxvYi5jb3JlLndpbmRvd3MubmV0L2NvY28yMDE0L3RyYWluMjAxNC56aXA=) 

</details>

<details>
    <summary>How to Download COCO-Text Annotations</summary>
    
The COCO-Text annotations (in JSON format) can be downloaded from [here](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=5&f=aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2NvY290ZXh0L0NPQ09fVGV4dC56aXA=)

</details>

After downloading the images and annotations, unzip the files, after which the data structure should be like as follows (ignoring the archive files):
```txt
COCO-Text
  |--- train_images
  |    |--- COCO_train2014_000000000036.jpg
  |    |--- COCO_train2014_000000000064.jpg
  |    |--- ...
  |--- COCO_Text.json
```

## Data Preparation

### For Detection Task

To prepare the data for text detection, you can run the following commands:

```bash
python tools/dataset_converters/convert.py \
    --dataset_name cocotext --task det \
    --image_dir path/to/COCO-Text/train_images/ \
    --label_dir path/to/COCO-Text/COCO_Text.json \
    --output_path path/to/COCO-Text/det_gt.txt
```

Then you can have an annotation file in the standard form `det_gt.txt` under the folder `COCO-Text/`.

[Back to README](../../../tools/dataset_converters/README.md)
