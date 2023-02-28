#!/bin/bash

atc --model=ch_ppocr_server_v2.0_det_infer.onnx \
	--framework=5 \
	--input_shape="x:1,3,-1,-1" \
	--input_format=ND \
	--dynamic_dims="1280,128;1280,256;1280,384;1280,512;1280,640;1280,768;1280,896;1280,1024;1280,1152;1280,1280;128,1280;256,1280;384,1280;512,1280;640,1280;768,1280;896,1280;1024,1280;1152,1280" \
	--soc_version=Ascend310P3 \
	--output=dbnet_dynamic_shape \
	--log=error