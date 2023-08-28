# CTW 数据集

## 数据集下载

COCO-Text数据集[官网](https://ctwdataset.github.io/) | [下载链接](https://ctwdataset.github.io/downloads.html)

> 注意: 您需要填写表格才能下载此数据集。

图像分为26批，即26个不同的.tar存档文件，格式为`images-trainval/ctw-trainval*.tar`。所有26批都需要下载。
注释存档文件名为`ctw-annotations.tar.gz`。

下载压缩后的图像后，解压后将所有图像收集到单个文件夹中，例如`train_val/`，注释也进行相应解压。最终目录结构如下：

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

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ctw --task det \
    --image_dir path/to/CTW/train_val/ \
    --label_dir path/to/CTW/train.jsonl \
    --output_path path/to/CTW/det_gt.txt
```

运行后，在文件夹`CTW/`下会生成注释文件`det_gt.txt`。

请注意，可以更改`label_dir`以准备验证集。

[返回dataset converters](converters.md)
