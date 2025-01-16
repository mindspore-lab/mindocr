## Training - MindOCR Models List

### Text Detection

| model  |dataset |bs | cards | F-score | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| dbnet_mobilenetv3  | icdar2015  | 10 | 1 | 77.28 | 100 | 100 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet18     | icdar2015  | 20 | 1 | 81.73 | 186 | 108 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet50     | icdar2015  | 10 | 1 | 85.05 | 133 | 75.2 | O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet++_resnet50   | icdar2015  | 32 | 1 | 86.74 | 571 | 56 | O0 | [mindocr_dbnet++](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| psenet_resnet152   | icdar2015  | 8 | 8 | 82.06 | 769.6| 83.16| O0 | [mindocr_psenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
| psenet_resnet50   | icdar2015  | 8 | 8 | 81.37 | 304.138 | 210.43 | O0 | [mindocr_psenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
| psenet_mobilenetv3   | icdar2015  | 8 | 8 | 70.56 | 173.604 | 368.66 | O0 | [mindocr_psenet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
|  east_mobilenetv3   | icdar2015  | 20 | 8 |  75.32  |   138   | 1185 | O0 | [mindocr_east](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   |
| east_resnet50      | icdar2015  | 20 | 8 | 84.87 | 256 | 625 | O0 | [mindocr_east](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   |

### Text Recognition

| model  |dataset |bs | cards | acc | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| svtr_tiny          | IC03,13,15,IIIT,etc | 512 | 4 | 90.23 | 459 | 4560 | O2 |  [mindocr_svtr](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 16 | 8 | 82.03 | 22.06 | 5802.71 | O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 64 | 8 | 84.45 | 76.48 | 6694.84 | O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 512 | 4 | 85.19 | 449 | 4561 | O2 |  [mindocr_rare](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   |
| visionlan_resnet45| IC03,13,15,IIIT,etc | 192| 4 | 90.61 | 417 | 1840 | O2 | [mindocr_visionlan](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan) |
| master_resnet31   | IC03,13,15,IIIT,etc | 512 | 4 | 90.37 | 747 | 2741 | O2 |  [mindocr_master](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/master)   |
| robustscanner_resnet31 | IC13,15,IIIT,SVT,etc | 256 | 4 | 87.86 |   825   |   310   | O0  |           [mindocr_robustscanner](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/robustscanner)            |
| abinet_resnet45   | IC03,13,15,IIIT,etc | 768 | 8 | 91.35 | 718 | 628.11 | O0 |  [mindocr_abinet](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/abinet)   |

### Text Direction Classification

| model  |dataset |bs | cards | acc | ms/step | fps | amp | config |
:-:     |   :-:   | :-: | :-: |  :-:   |  :-:    | :-:  |  :-: |  :-:    |
| mobilenetv3 | RCTW17,MTWI,LSVT | 256 | 4 | 94.59 | 172.9 | 5923.5 | O0 | [mindocr_mobilenetv3](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |
