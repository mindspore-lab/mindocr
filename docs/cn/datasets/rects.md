# ReCTS 数据集

## 数据集下载

ReCTS数据集[官网](https://rrc.cvc.uab.es/?ch=12) | [下载链接](https://rrc.cvc.uab.es/?ch=12&com=downloads)

> 注意: 在下载之前，请先注册一个账号。

请从上述网站下载数据并解压缩文件。解压文件后，数据结构应该是这样的：
```txt
ReCTS
  |--- img
  |    |--- <image_name>.jpg
  |    |--- <image_name>.jpg
  |    |--- ...
  |--- gt
  |    |--- <image_name>.json
  |    |--- <image_name>.json
  |    |--- ...
  |--- gt_unicode
  |    |--- <image_name>.json
  |    |--- <image_name>.json
  |    |--- ...
```

## 数据准备

### 检测任务

要准备用于文本检测的数据，您可以运行以下命令：

```bash
python tools/dataset_converters/convert.py \
    --dataset_name rects --task det \
    --image_dir path/to/ReCTS/img/ \
    --label_dir path/to/ReCTS/gt_unicode.json \
    --output_path path/to/ReCTS/det_gt.txt
```

运行后，在文件夹`ReCTS/`下会生成注释文件`det_gt.txt`。

[返回dataset converters](converters.md)
