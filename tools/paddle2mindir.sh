#!/bin/bash

usage() {
    echo -e "Usage"
    echo -e "    paddle2mindir.sh [-m=PPOCR_MODEL_NAME] \\"
    echo -e "                     [-p=SAVE_DIR] \\"
    echo -e "                   "
    echo -e "Description"
    echo -e "    PPOCR_MODEL_NAME: Name of support models. Supported models: 'ch_PP-OCRv4', 'ch_PP-OCRv4_server'"
    echo -e "    SAVE_DIR: folder to save downloaded ppocr models and converted mindir"
    exit -1
}

SAVE_DIR_=ppocr_models
for key in "$@"; do
    case $key in
        -m=*|--ppocr_model_name=*) PPOCR_MODEL_NAME_="${key#*=}";;
        -p=*|--save_dir=*) SAVE_DIR_="${key#*=}";;
        -h|--help) usage;;
    esac
done

# convert ppocr to mindir with dynamic shape
generate_dynamic_shape_config_file(){
    if [ $GENERATE_CONFIG_FILE == "True" ]; then
        echo "[acl_build_options]" > $CONFIG_FILE
        echo "input_format=NCHW" >> $CONFIG_FILE
        echo "input_shape_range=x:[-1,3,$DATA_SHAPE_H,$DATA_SHAPE_W]" >> $CONFIG_FILE
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
# path to save log
LOG_NAME="paddle2mindir.log"
# folder to save downloaded ppocr models and converted mindir
SAVE_DIR=${SAVE_DIR_}
# If generated config.txt，default True
GENERATE_CONFIG_FILE=True
# Models, supported: ["ch_PP-OCRv4", "ch_PP-OCRv4_server"]
PPOCR_MODEL_NAME=${PPOCR_MODEL_NAME_}
############

FILE_PATH=$(cd "$(dirname "$0")"; pwd)
logFile="$FILE_PATH/$LOG_NAME"
infoCmd=">> $logFile 2>&1"

pip3 install paddle2onnx==1.0.5
ppocr_path=$FILE_PATH/$SAVE_DIR
mkdir -p $ppocr_path/models
cd $ppocr_path/models

if [ "$PPOCR_MODEL_NAME" = "ch_PP-OCRv4" ]; then
    det_model="ch_PP-OCRv4_det_infer"
    rec_model="ch_PP-OCRv4_rec_infer"
    cls_model="ch_ppocr_mobile_v2.0_cls_infer"
elif [ "$PPOCR_MODEL_NAME" = "ch_PP-OCRv4_server" ]; then
    det_model="ch_PP-OCRv4_det_server_infer"
    rec_model="ch_PP-OCRv4_rec_server_infer"
    cls_model="ch_ppocr_mobile_v2.0_cls_infer"
else
    echo "$PPOCR_MODEL_NAME is not supported"
    exit
fi

### det model convertion
cd $ppocr_path/models
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/${det_model}.tar --no-check-certificate
tar xvf ${det_model}.tar
cd ${det_model}
SAVE_DB_ONNX_FILE=$ppocr_path/models/${det_model}/det_db.onnx
cmd="paddle2onnx --model_dir $ppocr_path/models/${det_model} --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file $SAVE_DB_ONNX_FILE --opset_version 11 --enable_onnx_checker True $infoCmd"
echo -e "\033[36mpaddle2onnx command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_ONNX_FILE=$SAVE_DB_ONNX_FILE
report_paddle2onnx
CONFIG_FILE=$ppocr_path/models/${det_model}/dynamic_config.txt
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
generate_dynamic_shape_config_file
SAVE_DB_MINDIR_FILE=$ppocr_path/models/${det_model}/det_db_dynamic_output
cmd="$CONVERTER_PATH --saveType=MINDIR --fmk=ONNX --optimize=ascend_oriented --modelFile=$SAVE_DB_ONNX_FILE --outputFile=$SAVE_DB_MINDIR_FILE --configFile=dynamic_config.txt $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_CONVERT_FILE=$SAVE_DB_MINDIR_FILE
report_convert
mv $ppocr_path/models/${det_model}/det_db_dynamic_output.mindir $ppocr_path/${PPOCR_MODEL_NAME}_det_db_dynamic_output.mindir

### rec model convertion
cd $ppocr_path/models
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/${rec_model}.tar --no-check-certificate
tar xvf ${rec_model}.tar
cd ${rec_model}
SAVE_REC_ONNX_FILE=$ppocr_path/models/${rec_model}/rec_crnn.onnx
cmd="paddle2onnx --model_dir $ppocr_path/models/${rec_model} --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file $SAVE_REC_ONNX_FILE --opset_version 11 --enable_onnx_checker True $infoCmd"
echo -e "\033[36mpaddle2onnx command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_ONNX_FILE=$SAVE_REC_ONNX_FILE
report_paddle2onnx
CONFIG_FILE=$ppocr_path/models/${rec_model}/dynamic_config.txt
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
generate_dynamic_shape_config_file
SAVE_REC_MINDIR_FILE=$ppocr_path/models/${rec_model}/rec_crnn_dynamic_output
cmd="$CONVERTER_PATH --saveType=MINDIR --fmk=ONNX --optimize=ascend_oriented --modelFile=$SAVE_REC_ONNX_FILE --outputFile=$SAVE_REC_MINDIR_FILE --configFile=dynamic_config.txt $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_CONVERT_FILE=$SAVE_REC_MINDIR_FILE
report_convert
mv $ppocr_path/models/${rec_model}/rec_crnn_dynamic_output.mindir $ppocr_path/${PPOCR_MODEL_NAME}_rec_crnn_dynamic_output.mindir

### cls model convertion
cd $ppocr_path/models
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/${cls_model}.tar --no-check-certificate
tar xvf ${cls_model}.tar
cd ${cls_model}
SAVE_CLS_ONNX_FILE=$ppocr_path/models/${cls_model}/cls_mv4.onnx
cmd="paddle2onnx --model_dir $ppocr_path/models/${cls_model} --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file $SAVE_CLS_ONNX_FILE --opset_version 11 --enable_onnx_checker True $infoCmd"
echo -e "\033[36mpaddle2onnx command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_ONNX_FILE=$SAVE_CLS_ONNX_FILE
report_paddle2onnx
CONFIG_FILE=$ppocr_path/models/${cls_model}/dynamic_config.txt
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
generate_dynamic_shape_config_file
SAVE_CLS_MINDIR_FILE=$ppocr_path/models/${cls_model}/cls_mv4_dynamic_output
cmd="$CONVERTER_PATH --saveType=MINDIR --fmk=ONNX --optimize=ascend_oriented --modelFile=$SAVE_CLS_ONNX_FILE --outputFile=$SAVE_CLS_MINDIR_FILE --configFile=dynamic_config.txt $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile
eval $cmd
SAVE_CONVERT_FILE=$SAVE_CLS_MINDIR_FILE
report_convert
mv $ppocr_path/models/${cls_model}/cls_mv4_dynamic_output.mindir $ppocr_path/${PPOCR_MODEL_NAME}_cls_mv4_dynamic_output.mindir
