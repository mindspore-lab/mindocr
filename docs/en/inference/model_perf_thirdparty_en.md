English | [中文](../../cn/inference/model_perf_thirdparty_cn.md)
## Third-party Model Inference Performance Table

This document will give the performance of the third-party inference model using the MindIR format after performing [model conversion](./convert_tutorial_en.md).

### 1. Text detection

| name |  model  | backbone | test data | recall | precision | f-score | source |
|:----:|:------:|:--------:|:--------:|:------:|:---------:|:-------:|:----:|
|  ch_pp_server_det_v2.0  | DB |  ResNet18_vd       | MLT17      | 0.3637       |  0.6340         |  0.4622    | PaddleOCR |
| ch_pp_det_OCRv3       | DB  | MobileNetV3 | MLT17  | 0.2557         | 0.5021          | 0.3389 | PaddleOCR |
| ch_pp_det_OCRv2 | DB  | MobileNetV3 | MLT17 | 0.3258 | 0.6318 | 0.4299 | PaddleOCR|
| ch_pp_mobile_det_v2.0_slim | DB  | MobileNetV3 | MLT17 | 0.2346 | 0.4868 | 0.3166 | PaddleOCR|
| ch_pp_mobile_det_v2.0 | DB  | MobileNetV3 | MLT17 | 0.2403 | 0.4597 | 0.3156 | PaddleOCR |
| en_pp_det_OCRv3 | DB  | MobileNetV3 | IC15 | 0.3866 | 0.4630 | 0.4214 | PaddleOCR |
| ml_pp_det_OCRv3 | DB  | MobileNetV3 | MLT17 | 0.5992 | 0.7348 | 0.6601 | PaddleOCR |

### 2. Text recognition
| name |  model  | backbone | test data | accuracy | norm edit distance | source |
|:----:|:------:|:--------:|:--------:|:------:|:---------:|:----:|
| ch_pp_server_rec_v2.0 | CRNN | ResNet34           | MLT17 (only Chinese) | 0.4991 | 0.7411 | PaddleOCR |
| ch_pp_rec_OCRv3       | SVTR | MobileNetV1Enhance | MLT17 (only Chinese) | 0.4991  | 0.7535 | PaddleOCR |
| ch_pp_rec_OCRv2       | CRNN | MobileNetV1Enhance | MLT17 (only Chinese) | 0.4459  | 0.7036     | PaddleOCR |
| ch_pp_mobile_rec_v2.0       | CRNN | MobileNetV3 | MLT17 (only Chinese) | 0.2459  | 0.4878        | PaddleOCR |
| en_pp_rec_OCRv3       | SVTR | MobileNetV1Enhance | MLT17 (only English) | 0.7964  | 0.8854        | PaddleOCR |
| en_pp_mobile_rec_number_v2.0_slim       | CRNN | MobileNetV3 | MLT17 (only English) | 0.0164  | 0.0657         | PaddleOCR |
| en_pp_mobile_rec_number_v2.0       | CRNN | MobileNetV3 | MLT17 (only English) | 0.4304  | 0.5944         | PaddleOCR |

Please note that the above models use model shape scaling, so the performance here only represents the performance under certain input shapes.

### 3. Evaluation method
Please refer to [Model Inference Evaluation](./model_evaluation_en.md) document.
