# RCTW-17 数据集

## 数据集下载

RCTW-17数据集[官网](https://rctw.vlrlab.net/) | [下载链接](https://rctw.vlrlab.net/dataset)

图像训练集分为两个集合`train_images.zip.001`和`train_images.zip.002`，注释文件为`*_gts.zip`。

图像下载解压缩后，请合并到同一个文件中，例如`train_images`，最终目录结构如下：
```txt
RCTW-17
  |--- train_images
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- train_gts
  |    |--- <image_name>.txt
  |    |--- <image_name>.txt
  |    |--- ...
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name rctw17 --task det \
    --image_dir path/to/RCTW-17/train_images/ \
    --label_dir path/to/RCTW-17/train_gts \
    --output_path path/to/RCTW-17/det_gt.txt
```

运行后，在文件夹`RCTW-17/`下会生成注释文件`det_gt.txt`。

[Back to dataset converters](converters.md)
