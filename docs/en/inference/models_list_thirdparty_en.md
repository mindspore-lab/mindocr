English | [中文](../../cn/inference/models_list_thirdparty_cn.md)

## Inference - Third-Party Models Support List

MindOCR can support the inference of third-party models, and this document displays a list of adapted models.

After downloading the model file, it needs to be converted to a model file supported by ACL/MindSpore Lite inference (OM
or MindIR). Please refer to the [model conversion tutorial](convert_tutorial_en.md).

The original model files involved are as follows:

| type     | format                                 | description                                                                                                                                                       |
|:---------|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pp-train | .pdparams、.pdopt、.states             | PaddlePaddle trained model, it saved in the training process, which stores the parameters of the model, mostly used for model evaluation and continuous training. |
| pp-infer | inference.pdmodel、inference.pdiparams | PaddlePaddle inference model, it can be derived from its trained model, saving the network structure and weights.                                                 |


### 1. Text detection

| name                  | model | backbone    | config                                                                           | download                                                                                       | reference                                                                                                        | source    |
|:----------------------|:------|:------------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------|
| ch_pp_server_det_v2.0 | DB    | ResNet18_vd | [yaml](../../../deploy/py_infer/src/configs/det/ppocr/ch_det_res18_db_v2.0.yaml) | [pp-infer](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) | [ch_ppocr_server_v2.0_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) | PaddleOCR |
| ch_pp_det_OCRv3       | DB    | MobileNetV3 | [yaml](../../../deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv3_det_cml.yaml)  | [pp-infer](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)         | [ch_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)          | PaddleOCR |

### 2. Text recognition

| name                  | model | backbone           | dict file                                                                                                     | config                                                                                    | download                                                                                       | reference                                                                                                        | source    |
|:----------------------|:------|:-------------------|:--------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------|
| ch_pp_server_rec_v2.0 | CRNN  | ResNet34           | [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt) | [yaml](../../../deploy/py_infer/src/configs/rec/ppocr/rec_chinese_common_train_v2.0.yaml) | [pp-infer](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) | [ch_ppocr_server_v2.0_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) | PaddleOCR |
| ch_pp_rec_OCRv3       | SVTR  | MobileNetV1Enhance | [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt) | [yaml](../../../deploy/py_infer/src/configs/rec/ppocr/ch_PP-OCRv3_rec_distillation.yaml)  | [pp-infer](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar)         | [ch_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)          | PaddleOCR |

### 3. Text angle classification

| name                  | model       | config                                                              | download                                                                                       | reference                                                                                                        | source    |
|:----------------------|:------------|:--------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------|
| ch_pp_mobile_cls_v2.0 | MobileNetV3 | [yaml](../../../deploy/py_infer/src/configs/cls/ppocr/cls_mv3.yaml) | [pp-infer](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) | PaddleOCR |
