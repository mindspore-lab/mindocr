English | [中文](../../cn/inference/models_list_cn.md)

## Inference - MindOCR Models Support List

MindOCR inference supports exported models from trained ckpt file, and this document displays a list of adapted models.

Please [export](../../../tools/export.py) or download the pre-exported MindIR file, and refer to the
[model conversion tutorial](./convert_tutorial_en.md) before inference.

### 1. Text detection

| model   | backbone    | language | config                                                       |
| :------ | :---------- | :------- | :----------------------------------------------------------- |
| DBNet   | MobileNetV3 | en       | [db_mobilenetv3_icdar15.yaml](../../../configs/det/dbnet/db_mobilenetv3_icdar15.yaml) |
|         | ResNet-18   | en       | [db_r18_icdar15.yaml](../../../configs/det/dbnet/db_r18_icdar15.yaml) |
|         | ResNet-50   | en       | [db_r50_icdar15.yaml](../../../configs/det/dbnet/db_r50_icdar15.yaml) |
| DBNet++ | ResNet-50   | en       | [db++_r50_icdar15.yaml](../../../configs/det/dbnet/db++_r50_icdar15.yaml) |
| EAST    | ResNet-50   | en       | [east_r50_icdar15.yaml](../../../configs/det/east/east_r50_icdar15.yaml) |
| PSENet  | ResNet-152  | en       | [pse_r152_icdar15.yaml](../../../configs/det/psenet/pse_r152_icdar15.yaml) |
|         | ResNet-152  | ch       | [pse_r152_ctw1500.yaml](../../../configs/det/psenet/pse_r152_ctw1500.yaml) |

### 2. Text recognition

| model | backbone    | dict file                                              | language | config                                                       |
| :---- | :---------- | :----------------------------------------------------- | :------- | :----------------------------------------------------------- |
| CRNN  | VGG7        | Default                                                | en       | [crnn_vgg7.yaml](../../../configs/rec/crnn/crnn_vgg7.yaml)   |
|       | ResNet34_vd | Default                                                | en       | [crnn_resnet34.yaml](../../../configs/rec/crnn/crnn_resnet34.yaml) |
|       | ResNet34_vd | [ch_dict.txt](../../../mindocr/utils/dict/ch_dict.txt) | ch       | [crnn_resnet34_ch.yaml](../../../configs/rec/crnn/crnn_resnet34_ch.yaml) |
