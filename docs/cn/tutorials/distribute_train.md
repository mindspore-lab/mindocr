# 分布式并行训练

本文档提供分布式并行训练的教程，在Ascend处理器上有两种方式可以进行单机多卡训练，通过OpenMPI运行脚本或通过配置RANK_TABLE_FILE进行单机多卡训练。在GPU处理器上可通过OpenMPI运行脚本进行单机多卡训练。

> 请确保在运行以下命令进行分布式训练之前，将 `yaml` 文件中的 `distribute` 参数设置为 `True`。

- [分布式并行训练](#分布式并行训练)
  - [1. Ascend](#1-ascend)
    - [1.1 通过OpenMPI运行脚本进行训练](#11-通过openmpi运行脚本进行训练)
    - [1.2 配置RANK\_TABLE\_FILE进行训练](#12-配置rank_table_file进行训练)
      - [1.2.1 使用八个（全部）设备进行训练](#121-使用八个全部设备进行训练)
      - [1.2.2 使用四个（部分）设备进行训练](#122-使用四个部分设备进行训练)
  - [2. GPU](#2-gpu)
    - [2.1 通过OpenMPI运行脚本进行训练](#21-通过openmpi运行脚本进行训练)

## 1. Ascend

**注**:

在Ascend平台， 有一些常见的使用限制，如下所示：

- 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*N卡设备集群。

- 每台机器的0-3卡和4-7卡各为1个组网，2卡和4卡训练时卡必须相连且不支持跨组网创建集群。也就是说，四卡训练时，只能选择`{0, 1, 2, 3}` 或者 `{4, 5, 6, 7}`. 进行双卡训练时, 跨组网的设备, 例如 `{0, 4}`不支持创建集群。 不过, 同一组网内的设备，例如 `{0, 1}`和 `{1, 2}`支持创建集群。



### 1.1 通过OpenMPI运行脚本进行训练

在 Ascend 硬件平台上，用户可以使用 `OpenMPI` 的 `mpirun` 命令来运行 `n` 个设备的分布式训练。例如，在 [DBNet Readme](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet/README_CN.md#34-training) 中，使用以下命令在设备 `0` 和设备 `1` 上训练模型：

```shell
# n 代表分布式训练使用的NPU的数量
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

> 请注意，`mpirun` 将从设备 `0` 开始按顺序在设备上运行训练。例如，`mpirun -n 4 python-command` 将在四个设备上运行训练：`{0, 1, 2, 3}`。


### 1.2 配置RANK_TABLE_FILE进行训练

#### 1.2.1 使用八个（全部）设备进行训练

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

之后运行以下命令即可：

``` shell
bash ascend8p.sh
```

以CRNN训练为例，其`ascend8p.sh`脚本为：
``` shell
#!/bin/bash
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE="./hccl_8p_01234567_127.0.0.1.json"

for ((i = 0; i < ${RANK_SIZE}; i++)); do
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
#### 1.2.2 使用四个（部分）设备进行训练

要在四个设备上运行训练，例如，`{4, 5, 6, 7}`，`RANK_TABLE_FILE`和运行脚本与在八个设备上运行使用的文件有所不同。

通过运行以下命令创建 `rank_table.json`：

```shell
python hccl_tools.py --device_num "[4,8)"
```

输出为：
```
hccl_4p_4567_127.0.0.1.json
```

其中， `hccl_4p_4567_127.0.0.1.json` 的示例为:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "127.0.0.1",
            "device": [
                {
                    "device_id": "4",
                    "device_ip": "192.168.100.100",
                    "rank_id": "0"
                },
                {
                    "device_id": "5",
                    "device_ip": "192.168.101.100",
                    "rank_id": "1"
                },
                {
                    "device_id": "6",
                    "device_ip": "192.168.102.100",
                    "rank_id": "2"
                },
                {
                    "device_id": "7",
                    "device_ip": "192.168.103.100",
                    "rank_id": "3"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

然后运行下面的命令可以开始训练:

```shell
bash ascend4p.sh
```

以CRNN训练为例，其`ascend4p.sh`脚本为：

```shell
#!/bin/bash
export DEVICE_NUM=8
export RANK_SIZE=4
export RANK_TABLE_FILE="./hccl_4p_4567_127.0.0.1.json"

for ((i = 0; i < ${RANK_SIZE}; i++)); do
    export DEVICE_ID=$((i+4))
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

注意， `DEVICE_ID` 和 `RANK_ID` 的组合关系应该跟 `hccl_4p_4567_127.0.0.1.json` 文件中相吻合.

## 2. GPU

### 2.1 通过OpenMPI运行脚本进行训练

在 GPU 硬件平台上，MindSpore也支持使用 `OpenMPI` 的 `mpirun` 命令来运行分布式训练。以下命令将在 `device 0`和 `device 1` 上运行训练。


```shell
# n 代表训练使用到的GPU数量
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

如果用户想在 `device 2` 和 `device 3` 上运行训练，用户可以在运行上面的命令之前运行 `export CUDA_VISIBLE_DEVICES=2,3`，或者直接运行以下命令：

```shell
# n 代表训练使用到的GPU数量
CUDA_VISIBLE_DEVICES=2,3 mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```
