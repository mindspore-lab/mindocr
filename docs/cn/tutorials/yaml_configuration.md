# 配置文件参数说明

- [环境参数-system](#1-环境参数-system)
- [共用参数-common](#2-共用参数-common)
- [模型定义参数-model](#3-模型定义参数-model)
- [后处理-postprocess](#4-后处理-postprocess)
- [评估指标-metric](#5-评估指标-metric)
- [损失函数-loss](#6-损失函数-loss)
- [学习率调整策略和优化器(scheduler, optimizer, loss_scaler)](#7-学习率调整策略和优化器-scheduler-optimizer-loss_scaler)
  - [学习率调整策略-scheduler](#学习率调整策略-scheduler)
  - [优化器-optimizer](#优化器-optimizer)
  - [损失缩放-loss_scaler](#损失缩放系数-loss_scaler)
- [训练和评估流程(train, eval)](#8-训练评估流程-train-eval)
  - [训练流程-train](#训练流程-train)
  - [评估流程-eval](#评估流程-eval)


本文档以 `configs/rec/crnn/crnn_icdar15.yaml` 为例，详细说明参数的用途。

## 1. 环境参数 (system)

| 字段 | 说明 | 默认值 | 可选值 | 备注 |
| ---- | ---- | ---- | ---- | ---- |
| mode | MindSpore运行模式(静态图/动态图) | 0 | 0 / 1 | 0: 表示在GRAPH_MODE模式中运行; 1: PYNATIVE_MODE模式 |
| distribute | 是否开启并行训练 | True | True / False | \ |
| device_id | 指定单卡训练时的卡id | 7 | 机器可用的卡的id | 该参数仅在distribute=False（单卡训练）和环境变量DEVICE_ID未设置时生效。单卡训练时，如该参数和环境变量DEVICE_ID均未设置，则默认使用0卡。 |
| amp_level | 混合精度模式 | O0 | O0/O1/O2/O3 | 'O0' - 不变化。<br> 'O1' - 将白名单内的Cell和运算转为float16精度，其余部分保持float32精度。<br> 'O2' - 将黑名单内的Cell和运算保持float32精度，其余部分转为float16精度。<br> 'O3' - 将网络全部转为float16精度。|
| seed | 随机种子 | 42 | Integer | \ |
| ckpt_save_policy | 模型权重保存策略 | top_k | "top_k" 或 "latest_k" | "top_k"表示保存前k个评估指标分数最高的checkpoint；"latest_k"表示保存最新的k个checkpoint。 `k`的数值通过`ckpt_max_keep`参数定义 |
| ckpt_max_keep | 最多保存的checkpoint数量 | 5 | Integer | \ |
| log_interval | log输出间隔(单位:step) | 100 | Integer | \ |
| val_while_train | 是否开启边训练边评估 | True | True/False | 如果值为True，请同步配置eval数据集 |
| val_start_epoch | 从第几个epoch开始跑评估 | 1 | Interger |  |
| val_interval   | 评估间隔(单位: epoch)   | 1 | Interger |  |
| drop_overflow_update | 当loss/梯度溢出时，是否放弃更新网络参数 | True | True/False | 如果值为True，则当出现溢出时，不会更新网络参数 |


## 2. 共用参数 (common)

因为同一个参数可能在不同的配置部分都需要重复利用，所以您可以在这个部分自定义一些通用的参数，以便管理。


## 3. 模型定义参数 (model)

在MindOCR中，模型的网络架构划分为 Transform, Backbone, Neck和Head四个模块。详细请参阅[文档](../../../mindocr/models/README.md)，以下是各部分的配置说明与例子。

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| type | 网络类型 | - | 目前支持 rec/det; rec表示识别任务，det表示检测任务 |
| pretrained |  指定预训练权重路径或url | null | 支持本地checkpoint文件路径或url |
| **transform:**| tranform模块配置 |  |  |
| name | 指定transform网络的名字 | - | 目前支持 STN_ON 变换 |
| **backbone:** | 骨干网络配置 ||
| name | 指定骨干网络类名或规格函数名 | - | 目前已定义的类有 rec_resnet34, rec_vgg7, SVTRNet and det_resnet18, det_resnet50, det_resnet152, det_mobilenet_v3。亦可自定义新的类别，请参照文档指示定义。 |
| pretrained | 是否加载预训练骨干权重 | False | 支持传入bool类型或str类型，若为True，则通过backbone py件中定义的url链接下载并加载默认权重。若传入str，可指定本地checkpoint路径或url路径进行加载。 |
| **neck:** | 配置网络Neck | |
| name | Neck类名 | - | 目前已定义的类有 RNNEncoder, DBFPN, EASTFPN 和 PSEFPN. 亦可自定义新的类别，请参照文档指示定义。 |
| hidden_size | RNN隐藏层单元数 | - | \ |
| **head:** | 设置网络预测头 | |
| name | Head类名 | - | 目前支持CTCHead, AttentionHead, DBHead, EASTHead 以及 PSEHead |
| weight_init | 设置权重初始化 | 'normal' | \ |
| bias_init | 设置权偏差初始化 | 'zeros' | \ |
| out_channels | 设置分类数 | - | \ |

> 注意：对于不同网络，backbone/neck/head模块可配置参数会有所不同，具体可配置参数由上表模块的`name`参数指定的类的__init__入参所决定 （如若指定下neck模块的name为DBFPN，由于DBFPN类初始化包括adaptive入参，则可在yaml中model.head层级下配置adaptive等参数。

参考例子: [DBNet](../../../configs/det/dbnet/db_r50_mlt2017.yaml), [CRNN](../../../configs/rec/crnn/crnn_icdar15.yaml)

## 4. 后处理 (postprocess)

代码位置请看： [mindocr/postprocess](../../../mindocr/postprocess/)

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| name | 后处理类名 | - | 目前支持 DBPostprocess, EASTPostprocess, PSEPostprocess, RecCTCLabelDecode 和 RecAttnLabelDecode |
| character_dict_path | 识别字典路径 | None | 若值为None, 则使用默认字典[0-9a-z] |
| use_space_char | 设置是否添加空格到字典 | False | True/False |

> 注意：对于不同后处理方法（由name指定），可配置的参数有所不同，并由后处理类的初始化方法__init__的入参所决定。

参考例子: [DBNet](../../../configs/det/dbnet/db_r50_mlt2017.yaml), [PSENet](../../../configs/det/psenet/pse_r152_icdar15.yaml)

## 5. 评估指标 (metric)

代码位置请看： [mindocr/metrics](../../../mindocr/metrics)

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| name | 评估指标类名 | - | 目前支持 RecMetric, DetMetric |
| main_indicator | 主要指标，用于最优模型的比较 | hmean | 识别任务使用acc，检测任务建议使用f-score |
| character_dict_path | 识别字典路径 | None | 若值为None, 则使用默认字典 "0123456789abcdefghijklmnopqrstuvwxyz" |
| ignore_space | 是否过滤空格 | True | True/False |
| print_flag | 是否打印log | False | 如设置True，则输出预测结果和标准答案等信息 |


## 6. 损失函数 (loss)

代码位置请看： [mindocr/losses](../../../mindocr/losses)

| 字段 | 用途 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| name | 损失函数类名 | - | 目前支持 L1BalancedCELoss, CTCLoss, AttentionLoss, PSEDiceLoss, EASTLoss and CrossEntropySmooth |
| pred_seq_len | 预测文本的长度 | 26 | 由网络架构决定 |
| max_label_len | 最长标签长度 | 25 | 数值应小于网络预测文本的长度 |
| batch_size | 单卡批量大小 | 32 | \ |

> 注意：对于不同损失函数（由name指定），可配置的参数有所不同，并由所选的损失函数的入参所决定。

## 7. 学习率调整策略和优化器 (scheduler, optimizer, loss_scaler)

### 学习率调整策略 (scheduler)

代码位置请看： [mindocr/scheduler](../../../mindocr/scheduler)

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| scheduler | 学习率调度器名字 | 'constant' | 目前支持 'constant', 'cosine_decay', 'step_decay', 'exponential_decay', 'polynomial_decay', 'multi_step_decay' |
| min_lr | 学习率最小值 | 1e-6 | 'cosine_decay'调整学习率的下限 |
| lr | 学习率 | 0.01 | |
| num_epochs | 总训练epoch数 | 200 | 整个训练的总epoch数 |
| warmup_epochs |训练学习率warmp阶段的epoch数 | 3 | 对于'cosine_decay'，`warmup_epochs`表示将学习率从0提升到`lr`的时期。 |
| decay_epochs | 训练学习率衰减阶段epoch数 | 10 | 对于'cosine_decay'，表示在`decay_epochs`内将 `lr` 衰减到 `min_lr`。对于'step_decay'，表示每经过`decay_epochs`轮，按`decay_rate`因子将 `lr` 衰减一次。 |


### 优化器 (optimizer)

代码位置请看： [mindocr/optim](../../../mindocr/optim)

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| opt | 优化器名 | 'adam' | 目前支持'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'lion', 'nadam', 'adan', 'rmsprop', 'adagrad', 'lamb'. |
| filter_bias_and_bn | 设置是否排除bias和batch norm的权重递减 | True | 如果为 True，则权重衰减将不适用于 BN 参数和 Conv 或 Dense 层中的偏差。 |
| momentum | 动量 | 0.9 | \ |
| weight_decay | 权重递减率 | 0 | 需要注意的是，weight decay可以是一个常量值，也可以是一个Cell。仅当应用动态权重衰减时，它才是 Cell。动态权重衰减类似于动态学习率，用户只需要以全局步长为输入自定义一个权重衰减时间表，在训练过程中，优化器调用WeightDecaySchedule实例获取当前步长的权重衰减值。 |
| nesterov | 是否使用 Nesterov 加速梯度 (NAG) 算法来更新梯度。 | False | True/False |


### 损失缩放系数 (loss_scaler)

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| type | loss缩放方法类型 | static | 目前支持 static, dynamic。常用于混合精度训练|
| loss_scale | loss缩放系数 | 1.0 | \ |
| scale_factor | 当使用dynamic loss scaler时，动态调整loss_scale的系数 | 2.0 | 在每个训练步骤中，当发生溢出时，损失缩放值会更新为 `loss_scale`/`scale_factor`。 |
| scale_window | 当使用dynamic loss scaler时，经过scale_window训练步未出现溢出时，将loss_scale放大scale_factor倍 | 1000 | 如果连续的`scale_window`步数没有溢出，损失将增加`loss_scale`*`scale_factor`缩放 |


## 8. 训练、评估流程 (train, eval)

训练流程的配置放在 `train` 底下，评估阶段的配置放在 `eval` 底下。注意，在模型训练的时候，若打开边训练边评估模式，即val_while_train=True时，则在每个epoch训练完毕后按照 `eval` 底下的配置运行一次评估。在非训练阶段，只运行模型评估的时候，只读取 `eval` 配置。


### 训练流程 (train)

| 字段 | 说明 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| ckpt_save_dir | 设置模型保存路径 | ./tmp_rec | \ |
| resume | 训练中断后恢复训练，可设定True/False，或指定需要加载恢复训练的ckpt路径 | False | 可指定True/False配置是否恢复训练，若True，则加载ckpt_save_dir目录下的resume_train.ckpt继续训练。也可以指定ckpt文件路径进行加载恢复训练。 |
| dataset_sink_mode | MindSpore数据下沉模式 | - | 如果设置True，则数据下沉至处理器，至少在每个epoch结束后才能返回数据 |
| gradient_accumulation_steps | 累积梯度的步数 | 1 | 每一步代表一次正向计算，梯度累计完成再进行一次反向修正 |
| clip_grad | 是否裁剪梯度 | False | 如果设置True，则将梯度裁剪成 `clip_norm` |
| clip_norm | 设置裁剪梯度的范数 | 1 | \ |
| ema | 是否启动EMA算法  | False | \ |
| ema_decay | EMA衰减率 | 0.9999 | \ |
| pred_cast_fp32 | 是否将logits的数据类型强制转换为fp32 | False | \ |
| **dataset** | 数据集配置 | 详细请参阅[Data文档](../../../mindocr/data/README.md) ||
| type | 数据集类型 | - | 目前支持 LMDBDataset, RecDataset 和 DetDataset |
| dataset_root | 数据集所在根目录 | None | Optional |
| data_dir | 数据集所在子目录 | - | 如果没有设置`dataset_root`，请将此设置成完整目录 |
| label_file | 数据集的标签文件路径 | - | 如果没有设置`dataset_root`，请将此设置成完整路径，否则只需设置子路径 |
| sample_ratio | 数据集抽样比率 | 1.0 | 若数值<1.0，则随机选取 |
| shuffle | 是否打乱数据顺序 | 在训练阶段为True，否则为False | True/False |
| transform_pipeline | 数据处理流程 | None | 详情请看 [transforms](../../../mindocr/data/transforms/README.md) |
| output_columns | 数据加载（data loader）最终需要输出的数据属性名称列表（给到网络/loss计算/后处理) (类型：列表），候选的数据属性名称由transform_pipeline所决定。 | None | 如果值为None，则输出所有列。以crnn为例，output_columns: \['image', 'text_seq'\]  |
| net_input_column_index | output_columns中，属于网络construct函数的输入项的索引 | [0] | \ |
| label_column_index | output_columns中，属于loss函数的输入项的索引 | [1] | \ |
| **loader** | 数据加载设置 ||
| shuffle | 每个epoch是否打乱数据顺序 | 在训练阶段为True，否则为False | True/False |
| batch_size | 单卡的批量大小 | - | \ |
| drop_remainder | 当数据总数不能除以batch_size时是否丢弃最后一批数据 | 在训练阶段为True，否则为False | True/False |
| max_rowsize | 指定在多进程之间复制数据时，共享内存分配的最大空间 | 64 | \ |
| num_workers | 指定 batch 操作的并发进程数/线程数 | n_cpus / n_devices - 2 | 该值应大于或等于2 |

参考例子: [DBNet](../../../configs/det/dbnet/db_r50_mlt2017.yaml), [CRNN](../../../configs/rec/crnn/crnn_icdar15.yaml)

### 评估流程 (eval)

`eval` 的参数与 `train` 基本一样，只补充说明几个额外的参数，其余的请参考上述`train`的参数说明。

| 字段 | 用途 | 默认值 | 备注 |
| :---- | :---- | :---- | :---- |
| ckpt_load_path | 设置模型加载路径 | - | \ |
| num_columns_of_labels | 设置数据集输出列中的标签数 | None | 默认假设图像 (data[1:]) 之后的列是标签。如果值不为None，即image(data[1:1+num_columns_of_labels])之后的num_columns_of_labels列是标签，其余列是附加信息，如image_path。 |
| drop_remainder | 当数据总数不能除以batch_size时是否丢弃最后一批数据 | 在训练阶段为True，否则为False | 在做模型评估时建议设置成False，若不能整除，mindocr会自动选择一个最大可整除的batch size |
