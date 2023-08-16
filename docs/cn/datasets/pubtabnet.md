# PubTabNet 数据集

## 数据集下载

PubTabNet数据集[官网](https://developer.ibm.com/exchanges/data/all/pubtabnet/) | [下载地址](https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz)

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
pubtabnet
  |--- train
  |    |--- PMC1064074_007_00.png
  |    |--- PMC1064076_003_00.png
  |    |--- ...
  |--- test
  |    |--- PMC1064127_003_00.png
  |    |--- PMC1065052_003_00.png
  |    |--- ...
  |--- val
  |    |--- PMC1064865_002_00.png
  |    |--- PMC1079806_002_00.png
  |    |--- ...
  |--- PubTabNet_2.0.0.jsonl
```

## 数据准备

### 表格识别任务

要准备用于表格识别的标注数据，您可以运行以下命令：

- 划分training set标注

```bash
python tools/dataset_converters/convert.py \
    --dataset_name pubtabnet --task table \
    --image_dir path/to/pubtabnet/train/ \
    --label_dir path/to/pubtabnet/PubTabNet_2.0.0.jsonl \
    --output_path path/to/pubtabnet/pubtab_train.jsonl \
    --split train
```

- 划分validation set标注

```bash
python tools/dataset_converters/convert.py \
    --dataset_name pubtabnet --task table \
    --image_dir path/to/pubtabnet/val/ \
    --label_dir path/to/pubtabnet/PubTabNet_2.0.0.jsonl \
    --output_path path/to/pubtabnet/pubtab_val.jsonl \
    --split val
```

- 注：原数据集未提供testing set标注

运行后，在文件夹`pubtabnet/`下会生成注释文件`pubtab_train.jsonl`和`pubtab_val.jsonl`。

[返回dataset converters](converters.md)
