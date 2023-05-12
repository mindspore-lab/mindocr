English | [中文](../../cn/inference/models_list_cn.md)

## Inference - MindOCR Models Support List

MindOCR inference supports exported models from trained ckpt file, and this document displays a list of adapted models.

Please download the pre-exported MindIR file, and refer to the [model conversion tutorial](./convert_tutorial_en.md)


### 1. Text detection

| name                       | model   | backbone  | config                                                   | reference & downlaod                            |
|:---------------------------|:--------|:----------|:---------------------------------------------------------|:------------------------------------------------|
| en_ms_det_dbnet_resnet50   | DBNet   | ResNet-50 | [yaml](../../../configs/det/dbnet/db_r50_icdar15.yaml)   | [dbnet](../../../configs/det/dbnet/README.md)   |
| en_ms_det_dbnetpp_resnet50 | DBNet++ | ResNet-50 | [yaml](../../../configs/det/dbnet/db++_r50_icdar15.yaml) | [dbnet++](../../../configs/det/dbnet/README.md) |

### 2. Text recognition

| name                    | model | backbone    | dict file | config                                               | reference & downlaod                        |
|:------------------------|:------|:------------|:----------|:-----------------------------------------------------|:--------------------------------------------|
| en_ms_rec_crnn_resnet34 | CRNN  | ResNet34_vd | Default   | [yaml](../../../configs/rec/crnn/crnn_resnet34.yaml) | [crnn](../../../configs/rec/crnn/README.md) |
