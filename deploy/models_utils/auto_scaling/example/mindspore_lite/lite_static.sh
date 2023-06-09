#!/bin/bash

converter_lite  --modelFile=/path/to/ch_PP-OCRv3_det_infer.onnx \
                --fmk=ONNX \
                --configFile=lite_static.txt \
                --saveType=MINDIR \
                --NoFusion=false \
                --device=Ascend \
                --outputFile=output
