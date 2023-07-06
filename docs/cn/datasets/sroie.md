# SROIE 数据集

## 数据集下载

SROIE数据集[官网](https://rrc.cvc.uab.es/?ch=13) | [下载链接](https://rrc.cvc.uab.es/?ch=13&com=downloads)

> 注意: 在下载之前，请先注册一个账号。

该数据集共3个任务：任务1为文本检测，任务2为OCR，任务3为关键信息提取。这里，我们仅下载和使用任务1数据集。

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：
```txt
SROIE
  |--- train
  |    |--- <image_name>.jpg
  |    |--- <image_name>.txt
  |    |--- <image_name>.jpg
  |    |--- <image_name>.txt
  |    |--- ...
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name sroie --task det \
    --image_dir path/to/SROIE/train/ \
    --label_dir path/to/SROIE/train \
    --output_path path/to/SROIE/det_gt.txt
```

运行后，在文件夹`SROIE/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
