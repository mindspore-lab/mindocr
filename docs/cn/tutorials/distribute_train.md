# 分布式并行训练

本文档提供分布式并行训练的教程，在Ascend处理器上有两种方式可以进行单机多卡训练，通过OpenMPI运行脚本或通过配置RANK_TABLE_FILE进行单机多卡训练。在GPU处理器上可通过OpenMPI运行脚本进行单机多卡训练。

## 通过OpenMPI运行脚本进行训练
当前MindSpore在Ascend上已经支持了通过OpenMPI的mpirun命令运行脚本，用户可参考[dbnet readme](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/README_CN.md#34-训练)进行训练，以下为命令用例。
运行命令前请确保yaml文件中的`distribute`参数为True。

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```
## 配置RANK_TABLE_FILE进行训练

使用此种方法在进行分布式训练前需要创建json格式的HCCL配置文件，即生成RANK_TABLE_FILE文件，以下为生成8卡相应配置文件命令，更具体信息及相应脚本参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明，
``` shell
python hccl_tools.py --device_num "[0,8)"
```
输出为：
```
hccl_8p_10234567_127.0.0.1.json
```
其中`hccl_8p_10234567_127.0.0.1.json`中内容示例为：
```
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "127.0.0.1",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.168.100.101",
                    "rank_id": "0"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.101.101",
                    "rank_id": "1"
                },
                {
                    "device_id": "2",
                    "device_ip": "192.168.102.101",
                    "rank_id": "2"
                },
                {
                    "device_id": "3",
                    "device_ip": "192.168.103.101",
                    "rank_id": "3"
                },
                {
                    "device_id": "4",
                    "device_ip": "192.168.100.100",
                    "rank_id": "4"
                },
                {
                    "device_id": "5",
                    "device_ip": "192.168.101.100",
                    "rank_id": "5"
                },
                {
                    "device_id": "6",
                    "device_ip": "192.168.102.100",
                    "rank_id": "6"
                },
                {
                    "device_id": "7",
                    "device_ip": "192.168.103.100",
                    "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

之后运行以下命令即可，运行命令前请确保yaml文件中的`distribute`参数为True。
``` shell
bash ascend8p.sh
```

以CRNN训练为例，其`ascend8p.sh`脚本为：
``` shell
#!/bin/bash
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE="./hccl_8p_01234567_127.0.0.1.json"

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "Launching rank: ${RANK_ID}, device: ${DEVICE_ID}"
    if [ $i -eq 0 ]; then
      echo 'i am 0'
      python -u tools/train.py --config configs/rec/crnn/crnn_resnet34_zh.yaml &> ./train.log &
    else
      echo 'not 0'
      python -u tools/train.py --config configs/rec/crnn/crnn_resnet34_zh.yaml &> /dev/null &
    fi
done
```

当需要训练其他模型时，只要将脚本中的yaml config文件路径替换即可，即`python -u tools/train.py --config path/to/model_config.yaml`

此时训练已经开始，可在`train.log`中查看训练日志。
