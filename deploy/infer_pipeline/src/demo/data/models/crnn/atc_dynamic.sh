#!/bin/bash

atc --model=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
	  --framework=5 \
	  --input_format=ND \
	  --input_shape_range="x:[1~500,3,32,32~4096]" \
	  --output=crnn_dynamic \
	  --soc_version=Ascend310P3 \
	  --log=error