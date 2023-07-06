# Born-Digital Images 数据集

## 数据集下载

原生数字图像数据集（Born-Digital Images）[官网](https://rrc.cvc.uab.es/?ch=1) | [下载链接](https://rrc.cvc.uab.es/?ch=1&com=downloads)

> 注意: 在下载之前，请先注册一个账号。

该数据集分为4个任务: 任务1为文本定位, 任务2为文本分割, 任务3为单词识别, 任务4为端到端文本检测识别。这里我们仅考虑下载使用任务1数据集。

下载图像和注释后，解压缩文件并根据需要重命名，例如`train_images`是图像，`train_labels` 是标签, 最终目录结构如下：
```txt
Born-Digital
  |--- train_images
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- train_labels
  |    |--- <image_name>.txt
  |    |--- <image_name>.txt
  |    |--- ...
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name borndigital --task det \
    --image_dir path/to/Born-Digital/train_images/ \
    --label_dir path/to/Born-Digital/train_labels \
    --output_path path/to/Born-Digital/det_gt.txt
```

运行后，在文件夹`Born-Digital/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
