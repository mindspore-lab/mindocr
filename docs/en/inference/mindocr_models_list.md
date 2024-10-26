## MindOCR Support Models List

**Note: All results is test on 310P3 with MindSpore2.2.14.**

### Text Detection

| Model | Backbone | Language | Datset | F-score(%) | FPS | Data Shape (NCHW) | Lite convert config txt | Configuration File | Download Link |
|---|---|---|---|---|---|---|---|---|---|
| [DBNet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) | MobileNetV3 | en | IC15 | 76.96 | 26.19 | (1,3,736,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_mobilenetv3_icdar15.yaml) | [mindir]() |
| | ResNet-18 | en | IC15 | 81.73 | 24.04 | (1,3,736,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r18_icdar15.yaml) | [mindir]() |
| | ResNet-50 | en | IC15 | 85.00 | 21.69 | (1,3,736,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r50_icdar15.yaml) | [mindir]() |
| | ResNet-50 | ch + en | 12 datasets | 83.41 | 21.69 | (1,3,736,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r50_icdar15.yaml) | [mindir]() |
| [DBNet++](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) | ResNet-50 | en | IC15 | 86.79 | 8.46 | (1,3,1152,2048) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/dbpp_r50_icdar15.yaml) | [mindir]() |
| | ResNet-50 | ch + en | 12 datasets | 84.30 | 8.46 | (1,3,1152,2048) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/dbpp_r50_icdar15.yaml) | [mindir]() |
| [EAST](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east) | ResNet-50 | en | IC15 | 86.86 | 6.72 | (1,3,720,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east/east_r50_icdar15.yaml) | [mindir]() |
| | MobileNetV3 | en | IC15 | 75.32 | 26.77 | (1,3,720,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/east/east_mobilenetv3_icdar15.yaml) | [mindir]() |
| [PSENet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | ResNet-152 | en | IC15 | 82.50 | 2.52 | (1,3,1472,2624) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_icdar15.yaml) | [mindir]() |
| | ResNet-50 | en | IC15 | 81.37 | 10.16 | (1,3,736,1312) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r50_icdar15.yaml) | [mindir]() |
| | MobileNetV3 | en | IC15 | 70.56 | 10.38 | (1,3,736,1312) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_mv3_icdar15.yaml) | [mindir]() |
| [FCENet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet) | ResNet50 | en | IC15 | 78.94 | 14.59 | (1,3,736,1280) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/fcenet/fce_icdar15.yaml) | [mindir]() |

### Text Recognition

| Model | Backbone | Character Dict | Dataset | Acc(%) | FPS | Data Shape (NCHW) | Lite convert config txt | Configuration File | Download Link |
|---|---|---|---|---|---|---|---|---|---|
| [CRNN](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn) | VGG7 | Default | IC15 | 66.01 | 465.64 | (1,3,32,100) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_vgg7.yaml) | [mindir]() |
| | ResNet34_vd | Default | IC15 | 69.67 | 397.29 | (1,3,32,100) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34.yaml) | [mindir]() |
| | ResNet34_vd | [ch_dict.txt](https://github.com/mindspore-lab/mindocr/tree/main/mindocr/utils/dict/ch_dict.txt) | / | / | / | (1,3,32,320) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34_ch.yaml) | [mindir]() |
| [SVTR](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr) | Tiny | Default | IC15 | 79.92 | 338.04 | (1,3,64,256) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/svtr/svtr_tiny.yaml) | [mindir]() |
| [Rare](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare) | ResNet34_vd | Default | IC15 | 69.47 | 273.23 | (1,3,32,100) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare/rare_resnet34.yaml) | [mindir]() |
| | ResNet34_vd | [ch_dict.txt](https://github.com/mindspore-lab/mindocr/tree/main/mindocr/utils/dict/ch_dict.txt) | / | / | / | (1,3,32,320) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/rare/rare_resnet34_ch.yaml) | [mindir]() |
| [RobustScanner](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/robustscanner) | ResNet-31 | [en_dict90.txt](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/utils/dict/en_dict90.txt) | IC15 | 73.71 | 22.30 | (1,3,48,160) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/robustscanner/robustscanner_resnet31.yaml) | [mindir]() |
| [VisionLAN](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/visionlan) | ResNet-45 | Default | IC15 | 80.07 | 321.37 | (1,3,64,256) | [yaml(LA)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LA.yaml) | [mindir(LA)]() |


### Text Direction Classification
| Model | Backbone | Dataset | Acc(%) | FPS | Data Shape (NCHW) | Lite convert config txt | Configuration File | Download Link |
|---|---|---|---|---|---|---|---|---|
| [MobileNetV3](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3) | MobileNetV3 | / | / | / | (1,3,48,192) | [config.txt]() | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3/cls_mv3.yaml) | [mindir]() |
