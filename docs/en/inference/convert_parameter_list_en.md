English | [中文](../../cn/inference/convert_parameter_list_cn.md)

## Inference - Model Conversion Parameters

For MindOCR inference, the types of model shape supported by detection, classification, and recognition tasks are shown
in the table below:

| task | shape                                                             |
|:-----|:------------------------------------------------------------------|
| det  | static shape, dynamic image_size, and batch_size=1                |
| cls  | static shape, dynamic batch_size                                  |
| rec  | static shape, dynamic image_size, dynamic batch_size + image_size |

When using `converter_lite`/`atc` tool, it is necessary to convert the model to the supported shape types mentioned
above.

If dynamic support is required, it is necessary to ensure that the model is exported as a dynamic version(paddle->onnx,
ckpt->mindir).


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
