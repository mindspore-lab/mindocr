#!/bin/bash

usage() {
    echo -e "Usage"
    echo -e "    temp_export.sh [-c MODEL_NAME_OR CONFIG] \\"
    echo -e "                   [-p LOCAL_CKPT_PATH] \\"
    echo -e "                   [-d SAVE_DIR] \\"
    echo -e "                   [-D IS_DYNAMIC_SHAPE] \\"
    echo -e "                   [-H DATA_SHAPE_H] \\"
    echo -e "                   [-W DATA_SHAPE_W] \\"
    echo -e "                   [-T MODEL_TYPE]"
    echo "Description"
    echo -e "    MODEL_NAME_OR CONFIG: Name of the model to be converted or the path to model YAML config file. Required."
    echo -e "    LOCAL_CKPT_PATH: Path to a local checkpoint. If set, export mindir by loading local ckpt. Otherwise, export mindir by downloading online ckpt."
    echo -e "    SAVE_DIR: Directory to save the exported mindir file."
    echo -e "    IS_DYNAMIC_SHAPE: Whether the export data shape is dynamic or static."
    echo -e "    DATA_SHAPE_H: H in data shape [H, W] for exporting mindir files. Required when arg \`is_dynamic_shape\` is False. It is recommended to be the same as the rescaled data shape in evaluation to get the best inference performance."
    echo -e "    MODEL_TYPE: Model type. Required when arg \`is_dynamic_shape\` is True. Choices=[\"det\", \"rec\", \"cls\"]."
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
        -h) usage;;
    esac
done

# Export Preprocessing
TEMP_VAR=$(basename $MODEL_NAME_OR_CONFIG)
MODEL_NAME=${TEMP_VAR%.*}
EXPORT_MINDIR_FILENAME=${MODEL_NAME}.mindir

FILE_PATH=$(cd "$(dirname "$0")"; pwd) 

cmd="python tools/export.py --model_name_or_config ${MODEL_NAME_OR_CONFIG} --save_dir $FILE_PATH/$SAVE_DIR"

## Add LOCAL_CKPT_PATH To cmd
if [ ! -z "$LOCAL_CKPT_PATH" ]; then
    cmd="$cmd --local_ckpt_path ${LOCAL_CKPT_PATH}"
fi

## Add IS_DYNAMIC_SHAPE To cmd
if [ "$IS_DYNAMIC_SHAPE" == True ]; then
    cmd="$cmd --is_dynamic_shape ${IS_DYNAMIC_SHAPE} --model_type ${MODEL_TYPE}"
else
    cmd="$cmd --is_dynamic_shape ${IS_DYNAMIC_SHAPE} --data_shape ${DATA_SHAPE_H} ${DATA_SHAPE_W}"
fi

echo "--- Begin Export ---"
# Run Export
echo "command: $cmd"
eval $cmd

if [ -f "$FILE_PATH/$SAVE_DIR/$EXPORT_MINDIR_FILENAME" ]; then
    echo "Success: $FILE_PATH/$SAVE_DIR/$EXPORT_MINDIR_FILENAME"
    echo "--- Export Complete ---" 
else 
    echo "Failed: $FILE_PATH/$SAVE_DIR/$EXPORT_MINDIR_FILENAME"
    echo "--- Export Failed ---"
fi