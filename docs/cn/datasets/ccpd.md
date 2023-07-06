# Chinese City Parking Dataset (CCPD) 2019 数据集

## 数据集下载

CCPD数据集[下载链接](https://github.com/detectRecog/CCPD)

该数据集被分为3个部分：训练集、验证集和测试集，每个集合的标签可在`splits`文件夹下发现。

图像的注释可在图像的文件名中找到，具体格式及描述可在[官网](https://github.com/detectRecog/CCPD#dataset-annotations)查阅。

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
CCPD2019
  |--- ccpd_base
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- ccpd_blur
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- ...
  |--- ...
  |--- ...
  |--- splits
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ccpd --task det \
    --image_dir path/to/CCPD2019/ccpd_base/ \
    --label_dir path/to/CCPD2019/splits/train.txt
    --output_path path/to/CCPD2019/det_gt.txt
```

运行后，在文件夹`CCPD2019/`下会生成注释文件`det_gt.txt`。


[返回dataset converters](converters.md)
