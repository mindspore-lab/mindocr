[English](../../en/datasets/ctw1500.md) | 中文

# SCUT-CTW1500 Datasets

## 数据下载
文本检测数据集（SCUT-CTW1500）[官网](https://github.com/Yuliang-Liu/Curve-Text-Detector)

[下载数据集](https://github.com/Yuliang-Liu/Curve-Text-Detector)

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
ctw1500
 ├── ctw1500_train_labels
 │   ├── 0001.xml 
 │   ├── 0002.xml
 │   ├── ...
 ├── gt_ctw_1500
 │   ├── 0001001.txt
 │   ├── 0001002.txt
 │   ├── ...
 ├── test_images
 │   ├── 1001.jpg
 │   ├── 1002.jpg
 │   ├── ...
 ├── train_images
 │   ├── 0001.jpg
 │   ├── 0002.jpg
 │   ├── ...
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name ctw1500 --task det \
    --image_dir path/to/ctw1500/train_images/ \
    --label_dir path/to/ctw1500/ctw_1500_train_labels \
    --output_path path/to/ctw1500/train_det_gt.txt 
```
```bash
python tools/dataset_converters/convert.py \
    --dataset_name ctw1500 --task det \
    --image_dir path/to/ctw1500/test_images/ \
    --label_dir path/to/ctw1500/gt_ctw_1500 \
    --output_path path/to/ctw1500/test_det_gt.txt 
```

运行后，在文件夹 `ctw1500/` 下有两个注释文件 `train_det_gt.txt` 和 `test_det_gt.txt`。
