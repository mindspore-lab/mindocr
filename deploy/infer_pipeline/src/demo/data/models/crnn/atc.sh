#!/bin/bash

atc --model=ch_ppocr_server_v2.0_rec_infer.onnx \
--framework=5 \
--input_format=ND \
--input_shape="x:-1,3,32,-1" \
--dynamic_dims="1,320;1,640;1,960;1,1280;1,1600;1,1920;1,2240;" \
--output=dynamic_dims/crnn_bs_1 \
--soc_version=Ascend310P3 \
--log=error

atc --model=ch_ppocr_server_v2.0_rec_infer.onnx \
--framework=5 \
--input_format=ND \
--input_shape="x:-1,3,32,-1" \
--dynamic_dims="4,320;4,640;4,960;4,1280;4,1600;4,1920;4,2240;" \
--output=dynamic_dims/crnn_bs_4 \
--soc_version=Ascend310P3 \
--log=error

atc --model=ch_ppocr_server_v2.0_rec_infer.onnx \
--framework=5 \
--input_format=ND \
--input_shape="x:-1,3,32,-1" \
--dynamic_dims="8,320;8,640;8,960;8,1280;8,1600;8,1920;8,2240;" \
--output=dynamic_dims/crnn_bs_8 \
--soc_version=Ascend310P3 \
--log=error

atc --model=ch_ppocr_server_v2.0_rec_infer.onnx \
--framework=5 \
--input_format=ND \
--input_shape="x:-1,3,32,-1" \
--dynamic_dims="16,320;16,640;16,960;16,1280;16,1600;16,1920;16,2240;" \
--output=dynamic_dims/crnn_bs_16 \
--soc_version=Ascend310P3 \
--log=error