# COCO-Text 数据集

## 数据集下载

COCO-Text数据集[官网](https://rrc.cvc.uab.es/?ch=5)

数据集图像和JSON标注文件`annotations v1.4 JSON`可从[此处](https://rrc.cvc.uab.es/?ch=5&com=downloads)下载。

> 注意：在下载之前，请先注册一个账号。

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
COCO-Text
  |--- train_images
  |    |--- COCO_train2014_000000000036.jpg
  |    |--- COCO_train2014_000000000064.jpg
  |    |--- ...
  |--- COCO_Text.json
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name cocotext --task det \
    --image_dir path/to/COCO-Text/train_images/ \
    --label_dir path/to/COCO-Text/COCO_Text.json \
    --output_path path/to/COCO-Text/det_gt.txt
```

运行后，在文件夹`COCO-Text/`下会生成注释文件`det_gt.txt`。

[返回ataset converters](converters.md)
