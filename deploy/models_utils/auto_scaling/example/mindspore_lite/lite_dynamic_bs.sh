#!/bin/bash

converter_lite  --modelFile=/path/to/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                --fmk=ONNX \
                --configFile=lite_dynamic_bs.txt \
                --saveType=MINDIR \
                --NoFusion=false \
                --device=Ascend \
                --outputFile=output
