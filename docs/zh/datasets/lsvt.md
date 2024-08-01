# LSVT 数据集

## 数据集下载

LSVT数据集[官网](https://rrc.cvc.uab.es/?ch=16) | [下载链接](https://rrc.cvc.uab.es/?ch=16&com=downloads)

> 注意: 在下载之前，请先注册一个账号。

图像需要下载`train_full_images_0.tar.gz`和`train_full_images_1.tar.gz`两个压缩文件，注释需要下载`train_full_labels.json`文件。

图像下载解压缩后，请合并到同一个文件中，例如`train_images`，最终目录结构如下：
```txt
LSVT
  |--- train_images
  |    |--- gt_0.jpg
  |    |--- gt_1.jpg
  |    |--- ...
  |--- train_full_labels.json
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name lsvt --task det \
    --image_dir path/to/LSVT/train_images/ \
    --label_dir path/to/LSVT/train_full_labels.json \
    --output_path path/to/LSVT/det_gt.txt
```

运行后，在文件夹`LSVT/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
