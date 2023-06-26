## 训练 - MindOCR模型支持列表

| model | type |dataset | fscore(detection)/accuracy(recognition) | mindocr recipe | vanilla mindspore
:-:     |  :-: |  :-:       | :-:        | :-:   | :-:
| dbnet_mobilenetv3  |detection| icdar2015          | 77.28 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet18     |detection| icdar2015          | 83.71 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet50     |detection| icdar2015          | 84.99 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/DBNet/)  |
| dbnet_resnet50     |detection| msra-td500         | 85.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet++_resnet50   |detection| icdar2015          | 86.60 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| psenet_resnet152   |detection| icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | [link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/psenet)   |
| east_resnet50      |detection| icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   | [link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/east)     |
| svtr_tiny          |recognition| IC03,13,15,IIIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |
| crnn_vgg7          |recognition| IC03,13,15,IIIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/CRNN)     |
| crnn_resnet34_vd   |recognition| IC03,13,15,IIIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| rare_resnet34_vd   |recognition| IC03,13,15,IIIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   |
