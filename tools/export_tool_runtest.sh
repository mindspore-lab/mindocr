#!/bin/bash

logFile="export_tool_runtest.log"
SAVE_DIR="tmprst"

infoCmd=">> $logFile 2>&1"
rm -f $logFile

report(){
    if [ -f "$SAVE_DIR/$EXPORT_MINDIR_FILENAME" ]; then
        echo -e "\033[32m Export Success \033[0m"
    else
        echo -e "\033[31m Export Failed \033[0m"
    fi
}

# --------------- Test Case 1 ---------------
## MODEL_NAME_OR_CONFIG is a valid File
MODEL_NAME_OR_CONFIG="configs/rec/crnn/crnn_icdar15.yaml"
LOCAL_CKPT_PATH="/root/.mindspore/models/crnn_resnet34-83f37f07.ckpt"

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=100
MODEL_TYPE=

cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo "TestCase: 1, cmd: $cmd"
eval $cmd

report

# --------------- Test Case 2 ---------------
## MODEL_NAME_OR_CONFIG is a valid String
MODEL_NAME_OR_CONFIG="crnn_resnet34"
LOCAL_CKPT_PATH="/root/.mindspore/models/crnn_resnet34-83f37f07.ckpt"

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=100
MODEL_TYPE=

cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo "TestCase: 2, cmd: $cmd"
eval $cmd

report
# --------------- Test Case 3 ---------------
## LOCAL_CKPT_PATH is a valid File
MODEL_NAME_OR_CONFIG="crnn_resnet34"
LOCAL_CKPT_PATH="/root/.mindspore/models/crnn_resnet34-83f37f07.ckpt"

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=100
MODEL_TYPE=

cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo "TestCase: 3, cmd: $cmd"
eval $cmd
report
# --------------- Test Case 4 ---------------
## LOCAL_CKPT_PATH is None
MODEL_NAME_OR_CONFIG="crnn_resnet34"
LOCAL_CKPT_PATH=

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=100
MODEL_TYPE=

cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo "TestCase: 4, cmd: $cmd"
eval $cmd
report


# --------------- Test Case 6 ---------------
## IS_DYNAMIC_SHAPE is True
MODEL_NAME_OR_CONFIG="crnn_resnet34"
LOCAL_CKPT_PATH=

IS_DYNAMIC_SHAPE=True
DATA_SHAPE_H=
DATA_SHAPE_W=
MODEL_TYPE="rec"

cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo "TestCase: 6, cmd: $cmd"
eval $cmd
report

# --------------- Test Case 7 ---------------
## IS_DYNAMIC_SHAPE is False
MODEL_NAME_OR_CONFIG="crnn_resnet34"
LOCAL_CKPT_PATH=

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=200
MODEL_TYPE=

cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
echo "TestCase: 7, cmd: $cmd"
eval $cmd
report

--------------- Test Case 8 ---------------
Convert all supported models with IS_DYNAMIC_SHAPE is False
LOCAL_CKPT_PATH=
IS_DYNAMIC_SHAPE=False
MODEL_TYPE=

models=('abinet' 'cls_mobilenet_v3_small_100_model' 'master_resnet31')
DATA_SHAPE_H=32
DATA_SHAPE_W=100
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('crnn_resnet34' 'crnn_resnet34_ch' 'crnn_vgg7')
DATA_SHAPE_H=32
DATA_SHAPE_W=100
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('dbnet_mobilenetv3' 'dbnet_resnet18' 'dbnet_resnet50')
DATA_SHAPE_H=736
DATA_SHAPE_W=1280
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('dbnetpp_resnet50')
DATA_SHAPE_H=1152
DATA_SHAPE_W=2048
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('east_mobilenetv3' 'east_resnet50')
DATA_SHAPE_H=720
DATA_SHAPE_W=1280
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('fcenet_resnet50')
DATA_SHAPE_H=736
DATA_SHAPE_W=1280
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('psenet_mobilenetv3' 'psenet_resnet50')
DATA_SHAPE_H=736
DATA_SHAPE_W=1312
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('psenet_resnet152')
DATA_SHAPE_H=1472
DATA_SHAPE_W=2624
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('rare_resnet34')
DATA_SHAPE_H=32
DATA_SHAPE_W=100
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('rare_resnet34_ch')
DATA_SHAPE_H=32
DATA_SHAPE_W=320
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('robustscanner_resnet31')
DATA_SHAPE_H=48
DATA_SHAPE_W=160
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

models=('svtr_tiny' 'svtr_tiny_ch' 'visionlan_resnet45')
DATA_SHAPE_H=64
DATA_SHAPE_W=256
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 8, models: $model, cmd: $cmd"
    eval $cmd
    report
done

# --------------- Test Case 9 ---------------
## Convert all supported models with IS_DYNAMIC_SHAPE is True
# rec
models=('abinet' 'crnn_resnet34' 'crnn_resnet34_ch' 'crnn_vgg7' 'master_resnet31' \
        'rare_resnet34' 'rare_resnet34_ch' 'robustscanner_resnet31' 'svtr_tiny' 'svtr_tiny_ch' 'visionlan_resnet45')
MODEL_TYPE="rec"

LOCAL_CKPT_PATH=
IS_DYNAMIC_SHAPE=True
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 9, models: $model, cmd: $cmd"
    eval $cmd
    report
done

# # det
models=('dbnet_mobilenetv3' 'dbnet_resnet18' 'dbnet_resnet50' \
        'dbnetpp_resnet50' 'east_mobilenetv3' 'east_resnet50' 'fcenet_resnet50' 'psenet_mobilenetv3' 'psenet_resnet152' 'psenet_resnet50')
MODEL_TYPE="det"

LOCAL_CKPT_PATH=
IS_DYNAMIC_SHAPE=True
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 9, models: $model, cmd: $cmd"
    eval $cmd
    report
done

# # cls
models=('cls_mobilenet_v3_small_100_model')
MODEL_TYPE="cls"

LOCAL_CKPT_PATH=
IS_DYNAMIC_SHAPE=True
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE $infoCmd"
    echo "TestCase: 9, models: $model, cmd: $cmd"
    eval $cmd
    report
done

