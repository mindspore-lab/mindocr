## Training - MindOCR Models List

### Text Detection

| model  |dataset | F-score | mindocr recipe |
:-:     |   :-:       | :-:        | :-:   |
| dbnet_mobilenetv3  | icdar2015          | 77.23 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet18     | icdar2015          | 81.73 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet50     | icdar2015          | 85.05 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet++_resnet50   | icdar2015          | 86.74 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| psenet_resnet152   | icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) |
| east_resnet50      | icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   |
| fcenet_resnet50    | icdar2015          | 84.12 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |

### Text Recognition

| model | dataset | acc | mindocr recipe |
:-:     |   :-:       | :-:        | :-:   |
| svtr_tiny          | IC03,13,15,IIIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   |

### Text Direction Classification

| model | dataset | acc | mindocr recipe |
:-:     |   :-:       | :-:        | :-:   |
| mobilenetv3          | RCTW17,MTWI,LSVT | 94.59 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |
