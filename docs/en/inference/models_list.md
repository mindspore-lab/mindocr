## Inference - MindOCR Models List

MindOCR inference supports exported models from trained ckpt file, and this document displays a list of adapted models.

Please [export](https://github.com/mindspore-lab/mindocr/blob/main/tools/export.py) or download the pre-exported MindIR
file, and refer to the [model conversion tutorial](convert_tutorial.md) before inference.

### 1. Text detection

| model   | backbone    | language | config                                                                                                                          |
|:--------|:------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------|
| DBNet   | MobileNetV3 | en       | [db_mobilenetv3_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_mobilenetv3_icdar15.yaml) |
|         | ResNet-18   | en       | [db_r18_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r18_icdar15.yaml)                 |
|         | ResNet-50   | en       | [db_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r50_icdar15.yaml)                 |
| DBNet++ | ResNet-50   | en       | [db++_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db++_r50_icdar15.yaml)             |
| EAST    | ResNet-50   | en       | [east_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east/east_r50_icdar15.yaml)              |
| PSENet  | ResNet-152  | en       | [pse_r152_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_icdar15.yaml)            |
|         | ResNet-152  | ch       | [pse_r152_ctw1500.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_ctw1500.yaml)            |

### 2. Text recognition

| model | backbone    | dict file                                                                                        | language | config                                                                                                             |
|:------|:------------|:-------------------------------------------------------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------|
| CRNN  | VGG7        | Default                                                                                          | en       | [crnn_vgg7.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_vgg7.yaml)               |
|       | ResNet34_vd | Default                                                                                          | en       | [crnn_resnet34.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34.yaml)       |
|       | ResNet34_vd | [ch_dict.txt](https://github.com/mindspore-lab/mindocr/tree/main/mindocr/utils/dict/ch_dict.txt) | ch       | [crnn_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34_ch.yaml) |
