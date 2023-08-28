## 推理 - 第三方模型
### 1. 第三方模型支持列表
MindOCR可以支持第三方模型（如PaddleOCR、MMOCR等）的推理，本文档展示了已适配的模型列表。 性能测试基于Ascend310P，部分模型暂无测试数据集。
#### 1.1 文本检测
|             名称             |  模型   |   骨干网络    | 数据集 | F-score(%) |  FPS  |    来源    |                                                                   配置文件                                                                   |                                                                                   下载                                                                                   |                                                          参考链接                                                          |
|:---------------------------:|:-------:|:-----------:|:-----:|:----------:|:-----:|:---------:|:-------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|    ch_pp_server_det_v2.0    |  DBNet  | ResNet18_vd | MLT17 |   46.22    | 21.65 | PaddleOCR |         [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_det_res18_db_v2.0.yaml)          |                                      [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar)                                       |   [ch_ppocr_server_v2.0_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)    |
|       ch_pp_det_OCRv3       |  DBNet  | MobileNetV3 | MLT17 |   33.89    | 22.40 | PaddleOCR |          [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv3_det_cml.yaml)          |                                          [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)                                           |        [ch_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)        |
|       ch_pp_det_OCRv2       |  DBNet  | MobileNetV3 | MLT17 |   42.99    | 21.90 | PaddleOCR |          [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv2_det_cml.yaml)          |                                          [weight](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar)                                           |        [ch_PP-OCRv2_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)        |
| ch_pp_mobile_det_v2.0_slim  |  DBNet  | MobileNetV3 | MLT17 |   31.66    | 19.88 | PaddleOCR |          [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_det_mv3_db_v2.0.yaml)           |                                  [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar)                                   | [ch_ppocr_mobile_slim_v2.0_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md) |
|    ch_pp_mobile_det_v2.0    |  DBNet  | MobileNetV3 | MLT17 |   31.56    | 21.96 | PaddleOCR |          [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_det_mv3_db_v2.0.yaml)           |                                      [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar)                                       |   [ch_ppocr_mobile_v2.0_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)    |
|       en_pp_det_OCRv3       |  DBNet  | MobileNetV3 | IC15  |   42.14    | 55.55 | PaddleOCR |          [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv3_det_cml.yaml)          |                                          [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar)                                           |        [en_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)        |
|       ml_pp_det_OCRv3       |  DBNet  | MobileNetV3 | MLT17 |   66.01    | 22.48 | PaddleOCR |          [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/ch_PP-OCRv3_det_cml.yaml)          |                                   [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar)                                   |        [ml_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)        |
| en_pp_det_dbnet_resnet50vd  |  DBNet  | ResNet50_vd | IC15  |   79.89    | 21.17 | PaddleOCR |             [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/det_r50_vd_db.yaml)             |                                         [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)                                          |          [DBNet](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_db_en.md)           |
| en_pp_det_psenet_resnet50vd |   PSE   | ResNet50_vd | IC15  |   80.44    | 7.75  | PaddleOCR |            [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/det_r50_vd_pse.yaml)             |                                       [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)                                       |         [PSE](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_psenet_en.md)          |
|  en_pp_det_east_resnet50vd  |  EAST   | ResNet50_vd | IC15  |   85.58    | 20.70 | PaddleOCR |            [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/det_r50_vd_east.yaml)            |                                        [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)                                         |          [EAST](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_east_en.md)          |
|  en_pp_det_sast_resnet50vd  |  SAST   | ResNet50_vd | IC15  |   81.77    | 22.14 | PaddleOCR |        [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/ppocr/det_r50_vd_sast_icdar15.yaml)        |                                    [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)                                     |          [SAST](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_sast_en.md)          |
| en_mm_det_dbnetpp_resnet50  | DBNet++ |  ResNet50   | IC15  |   81.36    | 10.66 |   MMOCR   | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/mmocr/dbnetpp_resnet50_fpnc_1200e_icdar2015.yaml) | [weight](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth) |                [DBNetpp](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnetpp/README.md)                |
|  en_mm_det_fcenet_resnet50  | FCENet  |  ResNet50   | IC15  |   83.67    | 3.34  |   MMOCR   |  [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/det/mmocr/fcenet_resnet50_fpn_1500e_icdar2015.yaml)  |   [weight](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015/fcenet_resnet50_fpn_1500e_icdar2015_20220826_140941-167d9042.pth)    |                 [FCENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md)                 |

**注意：在使用en_pp_det_psenet_resnet50vd模型进行推理时，需要使用以下命令修改onnx文件**

```shell
python deploy/models_utils/onnx_optim/insert_pse_postprocess.py \
      --model_path=./pse_r50vd.onnx \
      --binary_thresh=0.0 \
      --scale=1.0
```

#### 1.2 文本识别

|                名称                |  模型   |       骨干网络       |   数据集    | Acc(%) |  FPS   |    来源    |                                                         字典文件                                                          | 配置文件                                                                                                                                 | 下载                                                                                                                                                          | 参考链接                                                                                                                   |
|:---------------------------------:|:-------:|:------------------:|:----------:|:------:|:------:|:---------:|:------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|
|       ch_pp_server_rec_v2.0       |  CRNN   |      ResNet34      | MLT17 (ch) | 49.91  | 154.16 | PaddleOCR |      [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_chinese_common_train_v2.0.yaml)    | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar)                                                                 | [ch_ppocr_server_v2.0_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)       |
|          ch_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance | MLT17 (ch) | 49.91  | 408.38 | PaddleOCR |      [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/ch_PP-OCRv3_rec_distillation.yaml)     | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar)                                                                         | [ch_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)                |
|          ch_pp_rec_OCRv2          |  CRNN   | MobileNetV1Enhance | MLT17 (ch) | 44.59  | 203.34 | PaddleOCR |      [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/ch_PP-OCRv2_rec_distillation.yaml)     | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar)                                                                         | [ch_PP-OCRv2_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)                |
|       ch_pp_mobile_rec_v2.0       |  CRNN   |    MobileNetV3     | MLT17 (ch) | 24.59  | 167.67 | PaddleOCR |      [ppocr_keys_v1.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_chinese_lite_train_v2.0.yaml)      | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar)                                                                 | [ch_ppocr_mobile_v2.0_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)       |
|          en_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance | MLT17 (en) | 79.79  | 917.01 | PaddleOCR |            [en_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/en_dict.txt)             | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/en_PP-OCRv3_rec.yaml)                  | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar)                                                                         | [en_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)                |
| en_pp_mobile_rec_number_v2.0_slim |  CRNN   |    MobileNetV3     |     /      |   /    |   /    | PaddleOCR |            [en_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/en_dict.txt)             | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_en_number_lite_train.yaml)         | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar)                                                           | [en_number_mobile_slim_v2.0_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md) |
|   en_pp_mobile_rec_number_v2.0    |  CRNN   |    MobileNetV3     |     /      |   /    |   /    | PaddleOCR |            [en_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/en_dict.txt)             | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_en_number_lite_train.yaml)         | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar)                                                      | [en_number_mobile_v2.0_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)      |
|        korean_pp_rec_OCRv3        |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |      [korean_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/korean_dict.txt)      | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/korean_PP-OCRv3_rec.yaml)              | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar)                                                                | [korean_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)            |
|        japan_pp_rec_OCRv3         |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |       [japan_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/japan_dict.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/japan_PP-OCRv3_rec.yaml)               | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar)                                                                 | [japan_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)             |
|     chinese_cht_pp_rec_OCRv3      |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR | [chinese_cht_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/chinese_cht_dict.txt) | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/chinese_cht_PP-OCRv3_rec.yaml)         | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar)                                                           | [chinese_cht_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)       |
|          te_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |          [te_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/te_dict.txt)          | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/te_PP-OCRv3_rec.yaml)                  | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar)                                                                    | [te_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)                |
|          ka_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |          [ka_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/ka_dict.txt)          | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/ka_PP-OCRv3_rec.yaml)                  | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar)                                                                    | [ka_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)                |
|          ta_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |          [ta_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/ta_dict.txt)          | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/ta_PP-OCRv3_rec.yaml)                  | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar)                                                                    | [ta_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)                |
|        latin_pp_rec_OCRv3         |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |       [latin_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/latin_dict.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/latin_PP-OCRv3_rec.yaml)               | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar)                                                                 | [latin_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)             |
|        arabic_pp_rec_OCRv3        |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |      [arabic_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/arabic_dict.txt)      | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/arabic_PP-OCRv3_rec.yaml)              | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar)                                                                | [arabic_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)            |
|       cyrillic_pp_rec_OCRv3       |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |    [cyrillic_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/cyrillic_dict.txt)    | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/cyrillic_PP-OCRv3_rec.yaml)            | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar)                                                              | [cyrillic_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)          |
|      devanagari_pp_rec_OCRv3      |  SVTR   | MobileNetV1Enhance |     /      |   /    |   /    | PaddleOCR |  [devanagari_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/devanagari_dict.txt)  | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/devanagari_PP-OCRv3_rec.yaml)          | [weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar)                                                            | [devanagari_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)        |
|     en_pp_rec_crnn_resnet34vd     |  CRNN   |    ResNet34_vd     |    IC15    | 66.35  | 420.80 | PaddleOCR |          [ic15_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ic15_dict.txt)           | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_r34_vd_none_bilstm_ctc.yaml)       | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)                                                          | [CRNN](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6rc/doc/doc_en/algorithm_rec_crnn_en.md)                  |
|   en_pp_rec_rosetta_resnet34vd    | Rosetta |    Resnet34_vd     |    IC15    | 64.28  | 552.40 | PaddleOCR |          [ic15_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ic15_dict.txt)           | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_r34_vd_none_none_ctc.yaml)         | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)                                                            | [Rosetta](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_rec_rosetta_en.md)              |
|      en_pp_rec_vitstr_vitstr      | ViTSTR  |       ViTSTR       |    IC15    | 68.42  | 364.67 | PaddleOCR |     [EN_symbol_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/EN_symbol_dict.txt)      | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/ppocr/rec_vitstr_none_ce.yaml)               | [weight](https://paddleocr.bj.bcebos.com/rec_vitstr_none_ce_train.tar)                                                                                       | [ViTSTR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_rec_vitstr_en.md)                |
|      en_mm_rec_nrtr_resnet31      |  NRTR   |      ResNet31      |    IC15    | 67.26  | 32.63  |   MMOCR   |       [english_digits_symbols.txt](https://github.com/open-mmlab/mmocr/blob/main/dicts/english_digits_symbols.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/mmocr/nrtr_resnet31-1by8-1by4_6e_st_mj.yaml) | [weight](https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_resnet31-1by8-1by4_6e_st_mj/nrtr_resnet31-1by8-1by4_6e_st_mj_20220916_103322-a6a2a123.pth) | [NRTR](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/nrtr/README.md)                                    |
|    en_mm_rec_satrn_shallowcnn     |  SATRN  |     ShallowCNN     |    IC15    | 73.52  | 32.14  |   MMOCR   |       [english_digits_symbols.txt](https://github.com/open-mmlab/mmocr/blob/main/dicts/english_digits_symbols.txt)       | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/rec/mmocr/satrn_shallow_5e_st_mj.yaml)           | [weight](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_shallow_5e_st_mj/satrn_shallow_5e_st_mj_20220915_152443-5fd04a4c.pth)                    | [SATRN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/satrn/README.md)                                  |

#### 1.3 文本方向分类

|          名称          |    模型     | 数据集 | Acc(%) | FPS |    来源    |                                                    配置文件                                                     |                                             下载                                             |                                                       参考链接                                                        |
|:---------------------:|:-----------:|:-----:|:------:|:---:|:---------:|:-------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|
| ch_pp_mobile_cls_v2.0 | MobileNetV3 |   /   |   /    |  /  | PaddleOCR | [yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/py_infer/src/configs/cls/ppocr/cls_mv3.yaml) | [weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md) |

### 2. 第三方推理流程总览
```mermaid
graph LR;
    A[ThirdParty models] -- xx2onnx --> B[ONNX] -- converter_lite --> C[MindIR];
    C --input --> D[infer.py] -- outputs --> eval_rec.py/eval_det.py;
    H[images] --input --> D[infer.py];
```

### 3. 第三方模型推理方法
#### 3.1 文本检测

下面以[第三方模型支持列表](#11-文本检测)中的`en_pp_det_dbnet_resnet50vd`为例介绍推理方法：

- 下载第三方模型支持列表中的权重文件[weight](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)并解压；

- 由于该模型为paddle训练模型，需要先转换为推理模型（已为推理模型则跳过此步）：

```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python tools/export_model.py \
	-c configs/det/det_r50_vd_db.yml \
	-o Global.pretrained_model=./det_r50_vd_db_v2.0_train/best_accuracy  \
	Global.save_inference_dir=./det_db
```
执行完成后会生成以下内容：
``` text
det_db/
├── inference.pdmodel
├── inference.pdiparams
├── inference.pdiparams.info
```

- 下载并使用paddle2onnx工具(`pip install paddle2onnx`)，将推理模型转换成onnx文件：

```shell
paddle2onnx \
    --model_dir det_db \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file det_db.onnx \
    --opset_version 11 \
    --input_shape_dict="{'x':[-1,3,-1,-1]}" \
    --enable_onnx_checker True
```
paddle2onnx参数简要说明如下：

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 配置包含Paddle模型的目录路径|
|--model_filename |**[可选]** 配置位于 `--model_dir` 下存储网络结构的文件名|
|--params_filename |**[可选]** 配置位于 `--model_dir` 下存储模型参数的文件名称|
|--save_file | 指定转换后的模型保存目录路径 |
|--opset_version | **[可选]** 配置转换为ONNX的OpSet版本，目前支持 7~16 等多个版本，默认为 9 |
|--input_shape_dict | 输入Tensor的形状，用于生成动态ONNX模型，格式为 "{'x':[N,C,H,W]}"，-1表示动态shape |
|--enable_onnx_checker| **[可选]**  配置是否检查导出为 ONNX 模型的正确性， 建议打开此开关， 默认为 False|

参数中`--input_shape_dict`的值，可以通过[Netron](https://github.com/lutzroeder/netron)工具打开推理模型查看。
> 了解更多[paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop)

上述命令执行完成后会生成`det_db.onnx`文件;

- 在Ascend310/310P上使用converter_lite工具将onnx文件转换为mindir：

创建`config.txt`并指定模型输入shape，一个示例如下：
```
[ascend_context]
input_format=NCHW
input_shape=x:[1,3,736,1280]
```
配置文件参数简要说明如下：

|     参数     | 属性 |                                          功能描述                                         | 参数类型 |                     取值说明                    |
|:------------:|:----:|:-----------------------------------------------------------------------------------------:|:--------:|:-----------------------------------------------:|
| input_format | 可选 |                                     指定模型输入format                                    |  String  |            可选有"NCHW"、"NHWC"、"ND"           |
|  input_shape | 可选 | 指定模型输入Shape，input_name必须是转换前的网络模型中的输入名称，按输入次序排列，用；隔开 |  String  | 例如："input1:[1,64,64,3];input2:[1,256,256,3]" |
| dynamic_dims | 可选 |                             指定动态BatchSize和动态分辨率参数                             |  String  | 例如："dynamic_dims=[48,520],[48,320],[48,384]" |

> 了解更多[配置参数](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool_ascend.html)

执行以下命令：
```shell
converter_lite \
    --saveType=MINDIR \
    --fmk=ONNX \
    --optimize=ascend_oriented \
    --modelFile=det_db.onnx \
    --outputFile=det_db_output \
    --configFile=config.txt
```
上述命令执行完成后会生成`det_db_output.mindir`模型文件;

converter_lite参数简要说明如下：

|            参数           | 是否必选 |                            参数说明                            |             取值范围            | 默认值 |                       备注                       |
|:-------------------------:|:--------:|:--------------------------------------------------------------:|:-------------------------------:|:------:|:------------------------------------------------:|
|        fmk        |    是    |                       输入模型的原始格式                       | MINDIR、CAFFE、TFLITE、TF、ONNX |    -   |                         -                        |
|   saveType   |    否    |              设定导出的模型为mindir模型或者ms模型              |       MINDIR、MINDIR_LITE       | MINDIR | 云侧推理版本只有设置为MINDIR转出的模型才可以推理 |
|  modelFile  |    是    |                         输入模型的路径                         |                -                |    -   |                         -                        |
| outputFile |    是    |        输出模型的路径，不需加后缀，可自动生成.mindir后缀       |                -                |    -   |                         -                        |
| configFile |    否    | 1）可作为训练后量化配置文件路径；2）可作为扩展功能配置文件路径 |                -                |    -   |                         -                        |
|  optimize  |    否    |   设置针对设备的模型优化类型。若未设置，则不做优化  |  none、general、gpu_oriented、ascend_oriented  |    -   |                         -                        |


> 了解更多[converter_lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html)

> 了解更多[模型转换教程](convert_tutorial.md)



- 使用`/deploy/py_infer/infer.py`脚本和`det_db_output.mindir`文件执行推理：

```shell
python infer.py \
    --input_images_dir=/path/to/ic15/ch4_test_images \
    --det_model_path=/path/to/mindir/det_db_output.mindir \
    --det_model_name_or_config=en_pp_det_dbnet_resnet50vd \
    --res_save_dir=/path/to/dbnet_resnet50vd_results
```
执行完成后，在参数`--res_save_dir`所指目录下生成预测文件`det_results.txt`；

在进行推理时，可使用`--vis_det_save_dir`参数进行结果可视化：
<p align="center">
<img src="https://user-images.githubusercontent.com/15178426/253499854-ff5517f6-e8d0-493c-bc9a-8e384b2ac47a.jpg" width=60% />
</p>
<p align="center">
<em>文本检测结果可视化</em>
</p>

> 了解更多[infer.py](inference_tutorial.md#42-详细推理参数解释)推理参数

- 使用以下命令评估结果：
```shell
python deploy/eval_utils/eval_det.py \
		--gt_path=/path/to/ic15/test_det_gt.txt \
		--pred_path=/path/to/dbnet_resnet50vd_results/det_results.txt
```
结果为: `{'recall': 0.8281174771304767, 'precision': 0.7716464782413638, 'f-score': 0.7988852763585693}`
<br></br>
#### 3.2 文本识别

下面以[第三方模型支持列表](#12-文本识别)中的`en_pp_rec_OCRv3`为例介绍推理方法：

- 下载第三方模型支持列表中的权重文件[weight](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar)并解压；

- 由于该模型为paddle推理模型，直接进行第三步paddle转onnx（否则需要将训练模型转换为推理模型，参考上述文本检测）；

- 下载并使用paddle2onnx工具(`pip install paddle2onnx`)，将推理模型转换成onnx文件：

```shell
paddle2onnx \
    --model_dir en_PP-OCRv3_rec_infer \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file en_PP-OCRv3_rec_infer.onnx \
    --opset_version 11 \
    --input_shape_dict="{'x':[-1,3,48,-1]}" \
    --enable_onnx_checker True
```
paddle2onnx参数简要说明请见上述文本检测样例。

参数中`--input_shape_dict`的值，可以通过[Netron](https://github.com/lutzroeder/netron)工具打开推理模型查看。
> 了解更多[paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop)

参数中`--input_shape_dict`的值，可以通过[Netron](https://github.com/lutzroeder/netron)工具打开推理模型查看。

上述命令执行完成后会生成`en_PP-OCRv3_rec_infer.onnx`文件;

- 在Ascend310/310P上使用converter_lite工具将onnx文件转换为mindir：

创建`config.txt`并指定模型输入shape，一个示例如下：
```
[ascend_context]
input_format=NCHW
input_shape=x:[1,3,-1,-1]
dynamic_dims=[48,520],[48,320],[48,384],[48,360],[48,394],[48,321],[48,336],[48,368],[48,328],[48,685],[48,347]
```
配置参数简要说明请见上述文本检测样例。
> 了解更多[配置参数](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool_ascend.html)

执行以下命令：
```shell
converter_lite \
    --saveType=MINDIR \
    --fmk=ONNX \
    --optimize=ascend_oriented \
    --modelFile=en_PP-OCRv3_rec_infer.onnx \
    --outputFile=en_PP-OCRv3_rec_infer \
    --configFile=config.txt
```
上述命令执行完成后会生成`en_PP-OCRv3_rec_infer.mindir`模型文件；

converter_lite参数简要说明请见上述文本检测样例。
> 了解更多[converter_lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html)

> 了解更多[模型转换教程](convert_tutorial.md)

- 下载模型对应的字典文件[en_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/en_dict.txt)，使用`/deploy/py_infer/infer.py`脚本和`en_PP-OCRv3_rec_infer.mindir`文件执行推理：

```shell
python infer.py \
    --input_images_dir=/path/to/mlt17_en \
    --rec_model_path=/path/to/mindir/en_PP-OCRv3_rec_infer.mindir \
    --rec_model_name_or_config=en_pp_rec_OCRv3 \
    --character_dict_path=/path/to/en_dict.txt \
    --res_save_dir=/path/to/en_rec_infer_results
```
执行完成后，在参数`--res_save_dir`所指目录下生成预测文件`rec_results.txt`。
> 了解更多[infer.py](inference_tutorial.md#42-详细推理参数解释)推理参数

- 使用以下命令评估结果：

```shell
python deploy/eval_utils/eval_rec.py \
		--gt_path=/path/to/mlt17_en/english_gt.txt \
		--pred_path=/path/to/en_rec_infer_results/rec_results.txt
```
结果为: `{'acc': 0.7979344129562378, 'norm_edit_distance': 0.8859519958496094}`
<br></br>
