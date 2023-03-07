#!/bin/bash

atc --model=ch_ppocr_mobile_v2.0_cls_infer.onnx \
    --framework=5 \
    --input_shape="x:-1,3,48,192" \
    --input_format=ND \
    --dynamic_dims="1;4;8;16;32" \
    --soc_version=Ascend310P3 \
    --output=cls_310P \
    --log=error
