# TextOCR 数据集

## 数据集下载

TextOCR数据集[官网](https://textvqa.org/textocr/) | [下载链接](https://textvqa.org/textocr/dataset/)

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：
```txt
TextOCR
  |--- train_val_images
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- TextOCR_0.1_train.json
  |--- TextOCR_0.1_val.json
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name textocr --task det \
    --image_dir path/to/TextOCR/train_val_images/ \
    --label_dir path/to/TextOCR/TextOCR_0.1_train.json \
    --output_path path/to/TextOCR/det_gt.txt
```

运行后，在文件夹`TextOCR/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
