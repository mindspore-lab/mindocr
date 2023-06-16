[English](../../en/datasets/td500.md) | 中文

# MSRA Text Detection 500 Database (MSRA-TD500)

## 数据下载
文本检测数据集（MSRA-TD500）[官网](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))

[下载数据集](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
MSRA-TD500
 ├── test
 │   ├── IMG_0059.gt
 │   ├── IMG_0059.JPG
 │   ├── IMG_0080.gt
 │   ├── IMG_0080.JPG
 │   ├── ...
 ├── train
 │   ├── IMG_0030.gt
 │   ├── IMG_0030.JPG
 │   ├── IMG_0063.gt
 │   ├── IMG_0063.JPG
 │   ├── ...
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name td500 --task det \
    --image_dir path/to/MSRA-TD500/train/ \
    --label_dir path/to/MSRA-TD500/train \
    --output_path path/to/MSRA-TD500/train_det_gt.txt
```
```bash
python tools/dataset_converters/convert.py \
    --dataset_name td500 --task det \
    --image_dir path/to/MSRA-TD500/test/ \
    --label_dir path/to/MSRA-TD500/test \
    --output_path path/to/MSRA-TD500/test_det_gt.txt
```

运行后，在文件夹 `MSRA-TD500/` 下有两个注释文件 `train_det_gt.txt` 和 `test_det_gt.txt`。

[返回](../../../tools/dataset_converters/README_CN.md)
