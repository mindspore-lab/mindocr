#!/bin/bash

usage() {
    echo -e "Usage"
    echo -e "    temp2.sh [-c MODEL_NAME_OR CONFIG] \\"
    echo -e "                   [-p LOCAL_CKPT_PATH] \\"
    echo -e "                   [-d SAVE_DIR] \\"
    echo -e "                   [-D IS_DYNAMIC_SHAPE] \\"
    echo -e "                   [-H DATA_SHAPE_H] \\"
    echo -e "                   [-W DATA_SHAPE_W] \\"
    echo -e "                   [-T MODEL_TYPE] \\"
    echo -e "                   [-P CONVERTER_TOOL_PATH] \\"
    echo -e "                   [-O OPTIMIZE] \\"
    echo -e "                   [-C CONFIG_FILE] \\"
    echo -e "                   [-L Log_FILE] \\"
    echo -e "                   "
    echo -e "Description"
    echo -e "    MODEL_NAME_OR CONFIG: Name of the model to be converted or the path to model YAML config file. Required."
    echo -e "    LOCAL_CKPT_PATH: Path to a local checkpoint. If set, export mindir by loading local ckpt. Otherwise, export mindir by downloading online ckpt."
    echo -e "    SAVE_DIR: Directory to save the exported mindir file."
    echo -e "    IS_DYNAMIC_SHAPE: Whether the export data shape is dynamic or static."
    echo -e "    DATA_SHAPE_H: H in data shape [H, W] for exporting mindir files. Required when arg \`is_dynamic_shape\` is False. It is recommended to be the same as the rescaled data shape in evaluation to get the best inference performance."
    echo -e "    MODEL_TYPE: Model type. Required when arg \`is_dynamic_shape\` is True. Choices=[\"det\", \"rec\", \"cls\"]."
    echo -e "    CONVERTER_TOOL_PATH: Path to converter_lite tool. etc: mindspore/output/mindspore-lite-2.2.0-linux-x64/tools/converter"
    echo -e "    OPTIMIZE: Set the optimization accomplished in the process of converting model. Choices=[\"none\", \"general\", \"gpu_oriented\", \"ascend_oriented\"]"
    echo -e "    CONFIG_FILE: 1) Configure quantization parameter; 2) Profile path for extension."
    echo -e "    Log_FILE: 1) File saved log"
    exit -1
}


for key in "$@"; do
    case $key in
        -c=*|--model_name_or_config=*) MODEL_NAME_OR_CONFIG="${key#*=}";;
        -p=*|--local_ckpt_path=*) LOCAL_CKPT_PATH="${key#*=}";;
        -d=*|--save_dir=*) SAVE_DIR="${key#*=}";;
        -D=*|--is_dynamic_shape=*) IS_DYNAMIC_SHAPE="${key#*=}";;
        -H=*|--data_shape_h=*) DATA_SHAPE_H="${key#*=}";;
        -W=*|--data_shape_w=*) DATA_SHAPE_W="${key#*=}";;
        -T=*|--model_type=*) MODEL_TYPE="${key#*=}";;
        -P=*|--converter_path=*) CONVERTER_PATH="${key#*=}";;
        -O=*|--optimize=*) OPTIMIZE="${key#*=}";;
        -C=*|--config_file=*) CONFIG_FILE="${key#*=}";;
        -L=*|--log_file=*) LOG_FILE="${key#*=}";;
        -h) usage;;
    esac
done

logFile=$LOG_FILE
infoCmd=">> $logFile 2>&1"

report_export(){
    report_export_filename=$SAVE_DIR/$EXPORT_MINDIR_FILENAME
    if [ -f "$report_export_filename" ]; then
        echo -e "\033[32mExport Success\033[0m: $report_export_filename" | tee -a $logFile
    else
        echo -e "\033[31mExport Failed\033[0m: $report_export_filename" | tee -a $logFile
    fi
}

report_convert(){
    report_convert_filename="${SAVE_DIR}/${MODEL_NAME}_lite"
    # 2.1.1: ms, 2.2.0:mindir
    if [ -f "$report_convert_filename".ms ]; then
        echo -e "\033[32mConvert Success\033[0m: $report_convert_filename.ms" | tee -a $logFile
    elif [ -f "$report_convert_filename".mindir ]; then
        echo -e "\033[32mConvert Success\033[0m: $report_convert_filename.mindir" | tee -a $logFile
    else
        echo -e "\033[31mConvert Failed\033[0m: $report_convert_filename" | tee -a $logFile
    fi
}


cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo -e "\033[36mexport command:\033[0m $cmd"
eval $cmd
report_export

# converter_lite
FMK=MINDIR
MODEL_FILE="$SAVE_DIR/$MODEL_NAME.mindir"
OUTPUT_FILE="$SAVE_DIR/${MODEL_NAME}_lite"

if ! test -f $CONFIG_FILE ; then
    echo "CONFIG_FILE Not Found: "$CONFIG_FILE
    exit 1
fi

cmd="$CONVERTER_PATH/converter/converter_lite --fmk=$FMK --modelFile=$MODEL_FILE --outputFile=$OUTPUT_FILE"

if [ $OPTIMIZE ]; then
    cmd="$cmd --optimize=${OPTIMIZE}"
fi

# ConfigFile
if [ ! -z "$CONFIG_FILE" ]; then
    cmd="$cmd --configFile=${CONFIG_FILE}"
fi

# Run Export
cmd="$cmd $infoCmd"
echo -e "\033[36mconvert command:\033[0m $cmd" | tee -a $logFile    
eval $cmd
report_convert
echo "See $logFile for details."