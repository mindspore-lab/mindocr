# Distributed parallel training

This document provides a tutorial on distributed parallel training.
There are two ways to train on the Ascend AI processor: by running scripts with OpenMPI or configuring `RANK_TABLE_FILE` for training.
On GPU processors, scripts can be run with OpenMPI for training.

## Run scripts with OpenMPI

MindSpore supports scripts execution with OpenMPI's `mpirun` on Ascend hardware platform. Users can refer to [DBNet Readme](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/README.md#34-training) for more information on training. The following is the command use case:

```shell
# n is the number of GPUs/NPUs used in training
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

> Please ensure that the `distribute` parameter in the yaml file is set to `True` before running the command.

## Configure RANK_TABLE_FILE for training

Before using this method for distributed training, it is necessary to create an HCCL configuration file in json format,
i.e. generate RANK_TABLE_FILE. The following is the command to generate the corresponding configuration file for 8 devices
(for more information please refer to [HCCL tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)):

```shell
python hccl_tools.py --device_num "[0,8)"
```
This command produces the following output file:
```
hccl_8p_10234567_127.0.0.1.json
```

An example of the content in `hccl_8p_10234567_127.0.0.1.json`:

```json
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

Then start the training by running the following command:

```shell
bash ascend8p.sh
```

> Please ensure that the `distribute` parameter in the yaml file is set to `True` before running the command.

Here is an example of the `ascend8p.sh` script for CRNN training:

```shell
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
When training other models, simply replace the yaml config file path in the script, i.e. `path/to/model_config.yaml`.

After the training has started, and you can find the training log `train.log` in the project root directory.
