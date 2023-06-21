## 推理 - MindOCR模型支持列表

MindOCR推理支持训练端ckpt导出的模型，本文档展示了已适配的模型列表。

请[自行导出](https://github.com/mindspore-lab/mindocr/blob/main/tools/export.py)或下载已预先导出的MindIR文件，并参考[模型转换教程](convert_tutorial.md)，再进行推理。

### 1. 文本检测

| 模型     | 骨干网络     | 语言 | 配置文件                                                                                                                          |
|:--------|:------------|:----|:--------------------------------------------------------------------------------------------------------------------------------|
| DBNet   | MobileNetV3 | en  | [db_mobilenetv3_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_mobilenetv3_icdar15.yaml) |
|         | ResNet-18   | en  | [db_r18_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r18_icdar15.yaml)                 |
|         | ResNet-50   | en  | [db_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r50_icdar15.yaml)                 |
| DBNet++ | ResNet-50   | en  | [db++_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db++_r50_icdar15.yaml)             |
| EAST    | ResNet-50   | en  | [east_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east/east_r50_icdar15.yaml)              |
| PSENet  | ResNet-152  | en  | [pse_r152_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_icdar15.yaml)            |
|         | ResNet-152  | ch  | [pse_r152_ctw1500.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_ctw1500.yaml)            |

### 2. 文本识别

| 模型  | 骨干网络     | 字典文件                                                                                           | 语言 | 配置文件                                                                                                            |
|:-----|:------------|:-------------------------------------------------------------------------------------------------|:----|:-------------------------------------------------------------------------------------------------------------------|
| CRNN | VGG7        | Default                                                                                          | en  | [crnn_vgg7.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_vgg7.yaml)               |
|      | ResNet34_vd | Default                                                                                          | en  | [crnn_resnet34.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34.yaml)       |
|      | ResNet34_vd | [ch_dict.txt](https://github.com/mindspore-lab/mindocr/tree/main/mindocr/utils/dict/ch_dict.txt) | ch  | [crnn_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34_ch.yaml) |
