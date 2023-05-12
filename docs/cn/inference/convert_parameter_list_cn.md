[English](../../en/inference/convert_parameter_list_en.md) | 中文

## 推理 - 模型转换参数

MindOCR推理时，检测，分类和识别任务支持的模型Shape如下表：

| task | shape                                                             |
|:-----|:------------------------------------------------------------------|
| det  | static shape, dynamic image_size, and batch_size=1                |
| cls  | static shape, dynamic batch_size                                  |
| rec  | static shape, dynamic image_size, dynamic batch_size + image_size |

在使用`converter_lite`/`atc`模型转换时，需要将模型转换为上述支持的Shape类型。

如果需要动态支持，需要确保模型在导出（paddle->onnx, ckpt->mindir）时，导出为动态版的模型。


### 1. MindOCR

|            name            | type | ckpt->mindir<br/>shape |
|:--------------------------:|:----:|:----------------------:|
|  en_ms_det_dbnet_resnet50  | det  |     (1,3,736,1280)     |
| en_ms_det_dbnetpp_resnet50 | det  |    (1,3,1152,2048)     |
|  en_ms_rec_crnn_resnet34   | rec  |      (1,3,32,100)      |

### 2. PaddleOCR

|         name          | type | paddle->onnx<br>shape |   Lite   | ACL      |
|:---------------------:|:----:|:---------------------:|:--------:|:---------|
| ch_pp_server_det_v2.0 | det  |     (-1,3,-1,-1)      | &#10004; | &#10004; |
|     ch_pp_det_v3      | det  |     (-1,3,-1,-1)      | &#10004; | &#10004; |
| ch_pp_server_rec_v2.0 | rec  |     (-1,3,32,-1)      | &#10004; | &#10004; |
|     ch_pp_rec_v3      | rec  |     (-1,3,48,-1)      | &#10004; | &#10004; |
| ch_pp_mobile_cls_v2.0 | cls  |     (-1,3,48,192)     | &#10004; | &#10004; |
