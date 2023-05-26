[English](../../en/inference/models_list_thirdparty_en.md) | 中文

## 推理 - 第三方模型推理支持列表

MindOCR可以支持第三方模型的推理，本文档展示了已适配的模型列表。

在下载模型文件后，需要把它转换为ACL/MindSpore Lite推理支持的模型文件（MindIR或OM），请参考[模型转换教程](./convert_tutorial_cn.md)。

其中，涉及的原始模型文件如下表：

| 模型类型  | 模型格式                                 | 简介                                                                                    |
|:---------|:---------------------------------------|:---------------------------------------------------------------------------------------|
| pp-train | .pdparams、.pdopt、.states             | PaddlePaddle训练模型，训练过程中保存的模型的参数、优化器状态和训练中间信息，多用于模型指标评估和恢复训练 |
| pp-infer | inference.pdmodel、inference.pdiparams | PaddlePaddle推理模型，可由其训练模型导出得到，保存了模型的结构和参数                              |


### 1. 文本检测

| 名称                   | 模型 | 骨干网络     | 配置文件                                                                           | 下载                                                                                           | 参考链接                                                                                                           | 来源       |
|:----------------------|:----|:------------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------|
| ch_pp_server_det_v2.0 | DB  | ResNet18_vd | [yaml](../../../deploy/py_infer/src/configs/det/ppocr/ch_det_res18_db_v2.0.yaml) | [pp-infer](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) | [ch_ppocr_server_v2.0_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) | PaddleOCR |
| ch_pp_det_OCRv3       | DB  | MobileNetV3 | [yaml](../../../deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv3_det_cml.yaml)  | [pp-infer](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)         | [ch_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)          | PaddleOCR |

### 2. 文本识别

| 名称                   | 模型 | 骨干网络             | 字典文件                                                                                                       | 配置文件                                                                                    | 下载                                                                                            | 参考链接                                                                                                          | 来源       |
|:----------------------|:-----|:-------------------|:--------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------|
| ch_pp_server_rec_v2.0 | CRNN | ResNet34           | [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt) | [yaml](../../../deploy/py_infer/src/configs/rec/ppocr/rec_chinese_common_train_v2.0.yaml) | [pp-infer](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) | [ch_ppocr_server_v2.0_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) | PaddleOCR |
| ch_pp_rec_OCRv3       | SVTR | MobileNetV1Enhance | [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt) | [yaml](../../../deploy/py_infer/src/configs/rec/ppocr/ch_PP-OCRv3_rec_distillation.yaml)  | [pp-infer](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar)         | [ch_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)          | PaddleOCR |

### 3. 文本方向分类

| 名称                   | 模型        | 配置文件                                                              | 下载                                                                                            | 参考链接                                                                                                          | 来源       |
|:----------------------|:------------|:--------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------|
| ch_pp_mobile_cls_v2.0 | MobileNetV3 | [yaml](../../../deploy/py_infer/src/configs/cls/ppocr/cls_mv3.yaml) | [pp-infer](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) | PaddleOCR |
