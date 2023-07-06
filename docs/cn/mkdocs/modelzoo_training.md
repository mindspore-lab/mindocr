## 训练 - MindOCR模型支持列表

### 文本检测

| model  |dataset |bs | cards | F-score | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| dbnet_mobilenetv3  | icdar2015  | 10 | 1 | 77.23 | 100 | 100 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet18     | icdar2015  | 20 | 1 | 81.73 | 186 | 108 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet50     | icdar2015  | 10 | 1 | 85.05 | 133 | 75.2 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet++_resnet50   | icdar2015  | 32 | 1 | 86.74 | 571 | 56 | O0 | [mindocr_dbnet++](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| psenet_resnet152   | icdar2015  | 8 | 8 | 82.06 | 8455.88 | 7.57 | O0 | [mindocr_psenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
| east_resnet50      | icdar2015  | 20 | 8 | 84.87 | 256 | 625 | O0 | [mindocr_east](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   |
| fcenet_resnet50    | icdar2015  | 8 | 4 | 84.12 | 4570.64 | 7 | O0 | [mindocr_fcenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |

### 文本识别

| model  |dataset |bs | cards | acc | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| svtr_tiny          | IC03,13,15,IIIT,etc | 512 | 4 | 89.02 | 690 | 2968 | O2 |  [mindocr_svtr](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 16 | 8 | 82.03 | 22.06 | 5802.71 | O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 64 | 8 | 84.45 | 76.48 | 6694.84 | O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 512 | 4 | 85.19 | 449 | 4561 | O2 |  [mindocr_rare](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   |
| visionlan_resnet45| IC03,13,15,IIIT,etc | 192| 4 | 90.61 | 417 | 1840 | O2 | [mindocr_visionlan](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_1.yaml) |

### 文本方向分类

| model  |dataset |bs | cards | acc | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| mobilenetv3 | RCTW17,MTWI,LSVT | 256 | 4 | 94.59 | 172.9 | 5923.5 | O0 | [mindocr_mobilenetv3](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |
