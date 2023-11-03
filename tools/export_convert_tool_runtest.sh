#!/bin/bash

BASE_PATH=$(cd "$(dirname "$0")"; pwd) 

logFile="$BASE_PATH/export_convert_tool_runtest.log"
SAVE_DIR="tmprst"
CONVERTER_PATH="/home/z00579964/software/mindspore-lite-2.2.0-linux-x64/tools/converter"

infoCmd=">> $logFile 2>&1"
rm -f $logFile

# --------------- Test Case 1 ---------------
# 静态shape, optimize=general, config缺省
MODEL_NAME_OR_CONFIG="configs/rec/crnn/crnn_icdar15.yaml"
LOCAL_CKPT_PATH="/root/.mindspore/models/crnn_resnet34-83f37f07.ckpt"

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=100
MODEL_TYPE=

# OPTIMIZE="ascend_oriented"
OPTIMIZE="general"
CONFIG_FILE=

cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
log="TestCase 1, cmd: $cmd"
echo $log | tee -a $logFile
eval $cmd
echo ""

# --------------- Test Case 2 ---------------
# 静态shape, optimize=ascend_oriented, config缺省
MODEL_NAME_OR_CONFIG="configs/rec/crnn/crnn_icdar15.yaml"
LOCAL_CKPT_PATH="/root/.mindspore/models/crnn_resnet34-83f37f07.ckpt"

IS_DYNAMIC_SHAPE=False
DATA_SHAPE_H=32
DATA_SHAPE_W=100
MODEL_TYPE=

OPTIMIZE="ascend_oriented"
CONFIG_FILE=

cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
log="TestCase 2, cmd: $cmd"
echo $log | tee -a $logFile
eval $cmd
echo ""


# --------------- Test Case 3 ---------------
# 自有模型 静态shape转换
generate_static_shape_config_file(){
    if [ $GENERATE_CONFIG_FILE == "True" ]; then
        echo "[ascend_context]" > $CONFIG_FILE
        echo "input_format=NCHW" >> $CONFIG_FILE
        echo "input_shape=args0:[1,3,$DATA_SHAPE_H,$DATA_SHAPE_W]" >> $CONFIG_FILE
    fi
}

# Convert all supported models with IS_DYNAMIC_SHAPE is False
LOCAL_CKPT_PATH=
IS_DYNAMIC_SHAPE=False
MODEL_TYPE=

OPTIMIZE="ascend_oriented"
CONFIG_FILE="temp_config.txt"
GENERATE_CONFIG_FILE=True

infer_static_shape_ascend(){
    benchmark_cmd="benchmark --modelFile=$OUTPUT_FILE.mindir --device=Ascend --inputShapes='1,3,$DATA_SHAPE_H,$DATA_SHAPE_W' --loopCount=100 --warmUpLoopCount=10 $infoCmd"
    echo -e "\033[36mbenchmark command(Static shape):\033[0m $benchmark_cmd" | tee -a $logFile
    eval $benchmark_cmd
    if [ $? == 0 ]; then
        echo -e "\033[32mInfer Static Shape Success \033[0m" | tee -a $logFile
    else
        echo -e "\033[31mInfer Static Shape Failed \033[0m" | tee -a $logFile
    fi
}

# TODO: 转换有误
models=('abinet')
DATA_SHAPE_H=32
DATA_SHAPE_W=128
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('cls_mobilenet_v3_small_100_model' 'master_resnet31')
DATA_SHAPE_H=32
DATA_SHAPE_W=100
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('crnn_resnet34' 'crnn_resnet34_ch' 'crnn_vgg7')
DATA_SHAPE_H=32
DATA_SHAPE_W=100
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('dbnet_mobilenetv3' 'dbnet_resnet18' 'dbnet_resnet50')
DATA_SHAPE_H=736
DATA_SHAPE_W=1280
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('dbnetpp_resnet50')
DATA_SHAPE_H=1152
DATA_SHAPE_W=2048
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('east_mobilenetv3' 'east_resnet50')
DATA_SHAPE_H=720
DATA_SHAPE_W=1280
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('fcenet_resnet50')
DATA_SHAPE_H=736
DATA_SHAPE_W=1280
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('psenet_mobilenetv3' 'psenet_resnet50')
DATA_SHAPE_H=736
DATA_SHAPE_W=1312
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('psenet_resnet152')
DATA_SHAPE_H=1472
DATA_SHAPE_W=2624
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('rare_resnet34')
DATA_SHAPE_H=32
DATA_SHAPE_W=100
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('rare_resnet34_ch')
DATA_SHAPE_H=32
DATA_SHAPE_W=320
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

# TODO: 转换脚本不适配
models=('robustscanner_resnet31')
DATA_SHAPE_H=48
DATA_SHAPE_W=160
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=( 'svtr_tiny_ch' )
DATA_SHAPE_H=32
DATA_SHAPE_W=320
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('svtr_tiny' 'visionlan_resnet45')
DATA_SHAPE_H=64
DATA_SHAPE_W=256
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 3, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

# --------------- Test Case 4 ---------------
## 自有模型动态shape转换
generate_dynamic_shape_config_file(){
    if [ $GENERATE_CONFIG_FILE == "True" ]; then
        echo "[acl_build_options]" > $CONFIG_FILE
        echo "input_format=NCHW" >> $CONFIG_FILE
        echo "input_shape_range=args0:[1,3,$DATA_SHAPE_H,$DATA_SHAPE_W]" >> $CONFIG_FILE
    fi
}

infer_dynamic_shape_ascend(){
    benchmark_cmd="benchmark --modelFile=$OUTPUT_FILE.mindir --device=Ascend --inputShapes='1,3,$INFER_SHAPE_H,$INFER_SHAPE_W' --loopCount=100 --warmUpLoopCount=10 $infoCmd"
    echo -e "\033[36mbenchmark command(Dynamic shape):\033[0m $benchmark_cmd" | tee -a $logFile
    eval $benchmark_cmd
    if [ $? == 0 ]; then
        echo -e "\033[32mInfer Dynamic Shape Success \033[0m" | tee -a $logFile
    else
        echo -e "\033[31mInfer Dynamic Shape Failed \033[0m" | tee -a $logFile
    fi
}

LOCAL_CKPT_PATH=
IS_DYNAMIC_SHAPE=True

OPTIMIZE="ascend_oriented"
CONFIG_FILE="temp_config.txt"
GENERATE_CONFIG_FILE=True

models=('cls_mobilenet_v3_small_100_model')
DATA_SHAPE_H=32
DATA_SHAPE_W=-1
MODEL_TYPE=cls
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    infer_static_shape_ascend
    echo ""
done

models=('master_resnet31' 'abinet')
DATA_SHAPE_H=32
DATA_SHAPE_W=-1
MODEL_TYPE=rec
generate_static_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    INFER_SHAPE_H=32
    INFER_SHAPE_W=320
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=32
    INFER_SHAPE_W=352
    infer_dynamic_shape_ascend
    echo ""
done

models=('crnn_resnet34' 'crnn_resnet34_ch' 'crnn_vgg7')
DATA_SHAPE_H=32
DATA_SHAPE_W=-1
MODEL_TYPE=rec
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    # cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=general -C=$CONFIG_FILE -L=$logFile"
    
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    INFER_SHAPE_H=32
    INFER_SHAPE_W=320
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=32
    INFER_SHAPE_W=352
    infer_dynamic_shape_ascend
    echo ""
done

models=('dbnet_mobilenetv3' 'dbnet_resnet18' 'dbnet_resnet50' 'dbnetpp_resnet50')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=det
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    # Runing benchmark
    INFER_SHAPE_H=736
    INFER_SHAPE_W=1280
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=704
    INFER_SHAPE_W=1211
    infer_dynamic_shape_ascend
    echo ""
done

models=('dbnetpp_resnet50')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=det
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    # Runing benchmark
    INFER_SHAPE_H=1152
    INFER_SHAPE_W=2048
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=1120
    INFER_SHAPE_W=2080
    infer_dynamic_shape_ascend
    echo ""
done


models=('east_mobilenetv3' 'east_resnet50')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=det
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    # Runing benchmark
    INFER_SHAPE_H=720
    INFER_SHAPE_W=1280
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=816
    INFER_SHAPE_W=1211
    infer_dynamic_shape_ascend
    echo ""
done

models=('fcenet_resnet50')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=det
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    # Runing benchmark
    INFER_SHAPE_H=736
    INFER_SHAPE_W=1280
    infer_dynamic_shape_ascend
    echo ""
done

models=('psenet_mobilenetv3' 'psenet_resnet50')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=det
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    # Runing benchmark
    INFER_SHAPE_H=736
    INFER_SHAPE_W=1280
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=704
    INFER_SHAPE_W=1211
    infer_dynamic_shape_ascend
    echo ""
done

models=('psenet_resnet152')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=det
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    # Runing benchmark
    INFER_SHAPE_H=1472
    INFER_SHAPE_W=2624
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=1376
    INFER_SHAPE_W=1307
    infer_dynamic_shape_ascend
    echo ""
done

models=('rare_resnet34')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=rec
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd

    INFER_SHAPE_H=32
    INFER_SHAPE_W=100
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=32
    INFER_SHAPE_W=132
    infer_dynamic_shape_ascend
    echo ""
done

models=('rare_resnet34_ch')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=rec
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd

    INFER_SHAPE_H=32
    INFER_SHAPE_W=320
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=32
    INFER_SHAPE_W=352
    infer_dynamic_shape_ascend
    echo ""
done

# TODO: 转换脚本不适配
models=('robustscanner_resnet31')
DATA_SHAPE_H=48
DATA_SHAPE_W=160
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    echo ""
done

models=( 'svtr_tiny_ch' )
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=rec
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd

    INFER_SHAPE_H=32
    INFER_SHAPE_W=320
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=32
    INFER_SHAPE_W=352
    infer_dynamic_shape_ascend
    echo ""
done

models=('svtr_tiny' 'visionlan_resnet45')
DATA_SHAPE_H=-1
DATA_SHAPE_W=-1
MODEL_TYPE=rec
generate_dynamic_shape_config_file
for model in "${models[@]}"; do
    MODEL_NAME_OR_CONFIG=$model
    cmd="source export_convert_tool.sh -c=$MODEL_NAME_OR_CONFIG -p=$LOCAL_CKPT_PATH -d=$SAVE_DIR -D=$IS_DYNAMIC_SHAPE -H=$DATA_SHAPE_H -W=$DATA_SHAPE_W -T=$MODEL_TYPE -P=$CONVERTER_PATH -O=$OPTIMIZE -C=$CONFIG_FILE -L=$logFile"
    log="TestCase: 4, models: $model, cmd: $cmd"
    echo $log | tee -a $logFile
    eval $cmd
    
    INFER_SHAPE_H=64
    INFER_SHAPE_W=256
    infer_dynamic_shape_ascend

    INFER_SHAPE_H=64
    INFER_SHAPE_W=288
    infer_dynamic_shape_ascend
    echo ""
done