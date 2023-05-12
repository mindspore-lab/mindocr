[English](../../en/inference/models_list_en.md) | 中文

## 推理 - MindOCR模型推理支持列表


MindOCR推理支持训练端ckpt导出的模型，本文档展示了已适配的模型列表。

请在下载已预先导出的MindIR文件，并参考[模型转换教程](./convert_tutorial_cn.md)。


### 1. 文本检测

| 名称                        | 模型    | 骨干网络    | 配置文件                                                  | 参考&下载链接                                       |
|:---------------------------|:--------|:----------|:---------------------------------------------------------|:-------------------------------------------------|
| en_ms_det_dbnet_resnet50   | DBNet   | ResNet-50 | [yaml](../../../configs/det/dbnet/db_r50_icdar15.yaml)   | [dbnet](../../../configs/det/dbnet/README_CN.md) |
| en_ms_det_dbnetpp_resnet50 | DBNet++ | ResNet-50 | [yaml](../../../configs/det/dbnet/db++_r50_icdar15.yaml) | [dbnet++](../../../configs/det/dbnet/README_CN.md) |

### 2. 文本识别

| 名称                     | 模型 | 骨干网络      | 字典文件 | 配置文件                                              | 参考&下载链接                                     |
|:------------------------|:-----|:------------|:-------|:-----------------------------------------------------|:-----------------------------------------------|
| en_ms_rec_crnn_resnet34 | CRNN | ResNet34_vd | 默认字典 | [yaml](../../../configs/rec/crnn/crnn_resnet34.yaml) | [crnn](../../../configs/rec/crnn/README_CN.md) |
