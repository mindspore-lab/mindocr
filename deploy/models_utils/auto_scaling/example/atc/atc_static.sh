#!/bin/bash

atc --model=/path/to/ch_PP-OCRv3_det_infer.onnx \
    --framework=5 \
    --input_shape="x:1,3,1280,800" \
    --input_format=ND \
    --soc_version=Ascend310P3 \
    --output=output \
    --log=error
