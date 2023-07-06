# CASIA-10K 数据集

## 数据集下载

CASIA-10K 数据集[下载链接](http://www.nlpr.ia.ac.cn/pal/CASIA10K.html)

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
CASIA-10K
  |--- test
  |    |--- PAL00001.jpg
  |    |--- PAL00001.txt
  |    |--- PAL00005.jpg
  |    |--- PAL00005.txt
  |    |--- ...
  |--- train
  |    |--- PAL00003.jpg
  |    |--- PAL00003.txt
  |    |--- PAL00006.jpg
  |    |--- PAL00006.txt
  |    |--- ...
  |--- CASIA-10K_test.txt
  |--- CASIA-10K_train.txt
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name casia10k --task det \
    --image_dir path/to/CASIA-10K/train/ \
    --label_dir path/to/CASIA-10K/train \
    --output_path path/to/CASIA-10K/det_gt.txt
```

运行后，在文件夹`CASIA-10K/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
