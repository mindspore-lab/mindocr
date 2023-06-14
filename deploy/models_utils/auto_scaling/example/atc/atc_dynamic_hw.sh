#!/bin/bash

atc --model=ch_PP-OCRv3_det_infer.onnx \
    --framework=5 \
    --input_shape="x:1,3,-1,-1" \
    --input_format=ND \
    --dynamic_dims="1248,640;1248,672;1248,704;1248,736;1248,768;1248,800;1280,640;1280,672;1280,704;1280,736;1280,768;1280,800;" \
    --soc_version=Ascend310P3 \
    --output=output \
    --log=error
