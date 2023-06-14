[English](../../en/inference/model_perf_thirdparty_en.md) | 中文
## 第三方模型推理性能列表

本文档将给出第三方推理模型在进行[模型转换](./convert_tutorial_cn.md)后，使用MindIR格式推理时的性能。

### 1. 文本检测

| 名称 |  模型  | 骨干网络 | 测试数据 | recall | precision | f-score | 来源 |
|:----:|:------:|:--------:|:--------:|:------:|:---------:|:-------:|:----:|
|  ch_pp_server_det_v2.0  | DB |  ResNet18_vd       | MLT17      | 0.3637       |  0.6340         |  0.4622    | PaddleOCR |
| ch_pp_det_OCRv3       | DB  | MobileNetV3 | MLT17  | 0.2557         | 0.5021          | 0.3389 | PaddleOCR |
| ch_pp_det_OCRv2 | DB  | MobileNetV3 | MLT17 | 0.3258 | 0.6318 | 0.4299 | PaddleOCR|
| ch_pp_mobile_det_v2.0_slim | DB  | MobileNetV3 | MLT17 | 0.2346 | 0.4868 | 0.3166 | PaddleOCR|
| ch_pp_mobile_det_v2.0 | DB  | MobileNetV3 | MLT17 | 0.2403 | 0.4597 | 0.3156 | PaddleOCR |
| en_pp_det_OCRv3 | DB  | MobileNetV3 | IC15 | 0.3866 | 0.4630 | 0.4214 | PaddleOCR |
| ml_pp_det_OCRv3 | DB  | MobileNetV3 | MLT17 | 0.5992 | 0.7348 | 0.6601 | PaddleOCR |
| en_pp_det_sast_resnet50vd | SAST  | ResNet50_vd | IC15 | 0.7463 | 0.9043 | 0.8177 | PaddleOCR |
| en_mm_det_dbnetpp_resnet50 | DBNet++  | ResNet50 | IC15 | 0.8387 | 0.7900 | 0.8136 | MMOCR |
| en_mm_det_fcenet_resnet50 | FCENet  | ResNet50 | IC15 | 0.8681 | 0.8074 | 0.8367 | MMOCR |

### 2. 文本识别
| 名称 |  模型  | 骨干网络 | 测试数据 | accuracy | norm edit distance | 来源 |
|:----:|:------:|:--------:|:--------:|:------:|:---------:|:----:|
| ch_pp_server_rec_v2.0 | CRNN | ResNet34           | MLT17 (only Chinese) | 0.4991 | 0.7411 | PaddleOCR |
| ch_pp_rec_OCRv3       | SVTR | MobileNetV1Enhance | MLT17 (only Chinese) | 0.4991  | 0.7535 | PaddleOCR |
| ch_pp_rec_OCRv2       | CRNN | MobileNetV1Enhance | MLT17 (only Chinese) | 0.4459  | 0.7036     | PaddleOCR |
| ch_pp_mobile_rec_v2.0       | CRNN | MobileNetV3 | MLT17 (only Chinese) | 0.2459  | 0.4878        | PaddleOCR |
| en_pp_rec_OCRv3       | SVTR | MobileNetV1Enhance | MLT17 (only English) | 0.7964  | 0.8854        | PaddleOCR |
| en_pp_mobile_rec_number_v2.0_slim       | CRNN | MobileNetV3 | MLT17 (only English) | 0.0164  | 0.0657         | PaddleOCR |
| en_pp_mobile_rec_number_v2.0       | CRNN | MobileNetV3 | MLT17 (only English) | 0.4304  | 0.5944         | PaddleOCR |

请注意，上述模型采用了shape分档，因此该性能仅表示在某些shape下的性能。

### 3. 评估方法
请参考[模型推理精度评估](./model_evaluation_cn.md)文档。
