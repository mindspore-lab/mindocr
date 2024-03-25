#!/bin/bash
# convert ppocr to mindir with dynamic shape
generate_dynamic_shape_config_file(){
    if [ $GENERATE_CONFIG_FILE == "True" ]; then
        echo "[acl_build_options]" > $CONFIG_FILE
        echo "input_format=NCHW" >> $CONFIG_FILE
        echo "input_shape_range=x:[1,3,$DATA_SHAPE_H,$DATA_SHAPE_W]" >> $CONFIG_FILE
    fi
}
report_paddle2onnx(){
    report_paddle2onnx_filename=$SAVE_ONNX_FILE
    if [ -f "$report_paddle2onnx_filename" ]; then
        echo -e "\033[32mpaddle2onnx Success\033[0m: $report_paddle2onnx_filename" | tee -a $logFile
    else
        echo -e "\033[31mpaddle2onnx Failed\033[0m: $report_paddle2onnx_filename" | tee -a $logFile
    fi
}
report_convert(){
    report_convert_filename=$SAVE_CONVERT_FILE
    # 2.1.1: ms, 2.2.0:mindir
    if [ -f "$report_convert_filename".ms ]; then
        echo -e "\033[32mConvert Success\033[0m: $report_convert_filename.ms" | tee -a $logFile
    elif [ -f "$report_convert_filename".mindir ]; then
        echo -e "\033[32mConvert Success\033[0m: $report_convert_filename.mindir" | tee -a $logFile
    else
        echo -e "\033[31mConvert Failed\033[0m: $report_convert_filename" | tee -a $logFile
    fi
}
## pp-ocr configuration
# Converter_lite
CONVERTER_PATH="converter_lite"
# file to save log
logName="paddle2mindir.log"
# path to save converted ppocr models
SAVE_DIR="ppocr_models"
# if generate config.txt automatically，default True
GENERATE_CONFIG_FILE=True
############

tee -a $logFile
logFile="$FILE_PATH/$logName"
infoCmd=">> $logFile 2>&1"
FILE_PATH=$(cd "$(dirname "$0")"; pwd)
pip3 install paddle2onnx==1.0.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
ppocr_path=$FILE_PATH/$SAVE_DIR/ppocr
mkdir -p $ppocr_path/models
cd $ppocr_path/models
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar --no-check-certificate
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar --no-check-certificate

### det model convertion
tar xvf ch_PP-OCRv4_det_infer.tar
cd ch_PP-OCRv4_det_infer
SAVE_DB_ONNX_FILE=$ppocr_path/models/ch_PP-OCRv4_det_infer/det_db.onnx
cmd="paddle2onnx --model_dir $ppocr_path/models/ch_PP-OCRv4_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file $SAVE_DB_ONNX_FILE --opset_version 11 --enable_onnx_checker True $infoCmd"
echo -e "\033[36mpaddle2onnx command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_ONNX_FILE=$SAVE_DB_ONNX_FILE
report_paddle2onnx
CONFIG_FILE=$ppocr_path/models/ch_PP-OCRv4_det_infer/dynamic_config.txt
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
generate_dynamic_shape_config_file
SAVE_DB_MINDIR_FILE=$ppocr_path/models/ch_PP-OCRv4_det_infer/det_db_dynamic_output
cmd="$CONVERTER_PATH --saveType=MINDIR --fmk=ONNX --optimize=ascend_oriented --modelFile=$SAVE_DB_ONNX_FILE --outputFile=$SAVE_DB_MINDIR_FILE --configFile=dynamic_config.txt $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_CONVERT_FILE=$SAVE_DB_MINDIR_FILE
report_convert

### rec model convertion
cd $ppocr_path/models
tar xvf ch_PP-OCRv4_rec_infer.tar
cd ch_PP-OCRv4_rec_infer
SAVE_REC_ONNX_FILE=$ppocr_path/models/ch_PP-OCRv4_rec_infer/rec_crnn.onnx
cmd="paddle2onnx --model_dir $ppocr_path/models/ch_PP-OCRv4_rec_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file $SAVE_REC_ONNX_FILE --opset_version 11 --enable_onnx_checker True $infoCmd"
echo -e "\033[36mpaddle2onnx command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_ONNX_FILE=$SAVE_REC_ONNX_FILE
report_paddle2onnx
CONFIG_FILE=$ppocr_path/models/ch_PP-OCRv4_rec_infer/dynamic_config.txt
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
generate_dynamic_shape_config_file
SAVE_REC_MINDIR_FILE=$ppocr_path/models/ch_PP-OCRv4_rec_infer/rec_crnn_dynamic_output
cmd="$CONVERTER_PATH --saveType=MINDIR --fmk=ONNX --optimize=ascend_oriented --modelFile=$SAVE_REC_ONNX_FILE --outputFile=$SAVE_REC_MINDIR_FILE --configFile=dynamic_config.txt $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_CONVERT_FILE=$SAVE_REC_MINDIR_FILE
report_convert

### cls model convertion
cd $ppocr_path/models
tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar
cd ch_ppocr_mobile_v2.0_cls_infer
SAVE_CLS_ONNX_FILE=$ppocr_path/models/ch_ppocr_mobile_v2.0_cls_infer/cls_mv4.onnx
cmd="paddle2onnx --model_dir $ppocr_path/models/ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file $SAVE_CLS_ONNX_FILE --opset_version 11 --enable_onnx_checker True $infoCmd"
echo -e "\033[36mpaddle2onnx command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_ONNX_FILE=$SAVE_CLS_ONNX_FILE
report_paddle2onnx
CONFIG_FILE=$ppocr_path/models/ch_ppocr_mobile_v2.0_cls_infer/dynamic_config.txt
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
generate_dynamic_shape_config_file
SAVE_CLS_MINDIR_FILE=$ppocr_path/models/ch_ppocr_mobile_v2.0_cls_infer/cls_mv4_dynamic_output
cmd="$CONVERTER_PATH --saveType=MINDIR --fmk=ONNX --optimize=ascend_oriented --modelFile=$SAVE_CLS_ONNX_FILE --outputFile=$SAVE_CLS_MINDIR_FILE --configFile=dynamic_config.txt $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_CONVERT_FILE=$SAVE_CLS_MINDIR_FILE
report_convert

mv $ppocr_path/models/ch_PP-OCRv4_det_infer/det_db_dynamic_output.mindir ./
mv $ppocr_path/models/ch_PP-OCRv4_rec_infer/rec_crnn_dynamic_output.mindir ./
mv $ppocr_path/models/ch_ppocr_mobile_v2.0_cls_infer/cls_mv4_dynamic_output.mindir ./
