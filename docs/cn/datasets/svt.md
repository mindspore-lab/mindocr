# The Street View Text Dataset (SVT)

## 数据下载
街景文本数据集（SVT）[官网](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)

[下载数据集](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：

```txt
svt1
 ├── img
 │   ├── 00_00.jpg
 │   ├── 00_01.jpg
 │   ├── 00_02.jpg
 │   ├── 00_03.jpg
 │   ├── ...
 ├── test.xml
 └── train.xml
```

## 数据准备

### 识别任务

要准备用于文本识别的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name  svt --task rec \
    --image_dir path/to/svt1/ \
    --label_dir path/to/svt1/train.xml \
    --output_path path/to/svt1/rec_train_gt.txt
```

运行后，在文件夹 `svt1/` 下有一个文件夹 `cropped_images/` 和一个注释文件 `rec_train_gt.txt`。

[返回dataset converters](converters.md)
