#!/bin/bash
pushd ../deploy/models_utils/conversion/onnx_optim
python insert_argmax.py --model_path=/xx/ch_ppocr_server_v2.0_rec_infer.onnx \
                        --check_output_onnx=True &
pid1=$!
wait $pid1
popd

pushd ../deploy/models_utils/conversion/auto_gear
python auto_gear.py --image_path=/xx/lsvt/images \
                    --gt_path=/xx/lsvt/labels \
                    --det_onnx_path=ch_ppocr_server_v2.0_det_infer.onnx \
                    --rec_onnx_path=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
                    --rec_model_height=32 \
                    --soc_version=Ascend310P \
                    --output_path=./lsvt_om_v2 &&
python auto_select.py --rec_model_path=lsvt_om_v2/crnn
popd