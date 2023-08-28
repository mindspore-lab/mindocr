# ICDAR2019 ArT 数据集

## 数据集下载

ICDAR2019 ArT数据集 [官网](https://rrc.cvc.uab.es/?ch=14) | [下载链接](https://rrc.cvc.uab.es/?ch=14&com=downloads)
> 注意: 在下载之前，请先注册一个账号。

图像需要下载“任务1和任务3”部分中的存档文件`train_images.tar.gz`。注释需要下载同一节中的.JSON文件`train_labels.json`。

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：
```txt
ICDAR2019-ArT
  |--- train_images
  |    |--- train_images
  |    |    |--- gt_0.jpg
  |    |    |--- gt_1.jpg
  |    |    |--- ...
  |--- train_labels.json
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ic19_art --task det \
    --image_dir path/to/ICDAR2019-ArT/train_images/train_images/ \
    --label_dir path/to/ICDAR2019-ArT/train_labels.json \
    --output_path path/to/ICDAR2019-ArT/det_gt.txt
```

运行后，在文件夹`ICDAR2019-ArT/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
