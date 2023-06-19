# Configuration parameter description

- [system](#1-environment-parameters-system)
- [common](#2-shared-parameters-common)
- [model](#3-model-architecture-model)
- [postprocess](#4-postprocessing-postprocess)
- [metric](#5-evaluation-metrics-metric)
- [loss](#6-loss-function-loss)
- [scheduler, optimizer, loss_scaler](#7-learning-rate-adjustment-strategy-and-optimizer-scheduler-optimizer-loss_scaler)
  - [scheduler](#learning-rate-adjustment-strategy-scheduler)
  - [optimizer](#optimizer)
  - [loss_scaler](#loss-scaling-loss_scaler)
- [train, eval](#8-training-evaluation-and-predict-process-train-eval-predict)
  - [train](#training-process-train)
  - [eval](#evaluation-process-eval)

This document takes `configs/rec/crnn/crnn_icdar15.yaml` as an example to describe the usage of parameters in detail.

## 1. Environment parameters (system)

| Parameter | Description | Default | Optional Values ​​| Remarks |
| ---- | ---- | ---- | ---- | ---- |
| mode | Mindspore running mode (static graph/dynamic graph) | 0 | 0 / 1 | 0: means running in GRAPH_MODE mode; 1: PYNATIVE_MODE mode |
| distribute | Whether to enable parallel training | True | True / False | \ |
| device_id | Specify the device id while standalone training | 7 | The ids of all devices in the server | Only valid when distribute=False (standalone training) and environment variable 'DEVICE_ID' is NOT set. While standalone training, if both this arg and environment variable 'DEVICE_ID' are NOT set, use device 0 by default. |
| amp_level | Mixed precision mode | O0 | O0/O1/O2/O3 | 'O0' - no change. <br> 'O1' - convert the cells and operations in the whitelist to float16 precision, and keep the rest in float32 precision. <br> 'O2' - Keep the cells and operations in the blacklist with float32 precision, and convert the rest to float16 precision. <br> 'O3' - Convert all networks to float16 precision. |
| seed | Random seed | 42 | Integer | \ |
| ckpt_save_policy | The policy for saving model weights | top_k | "top_k" or "latest_k" | "top_k" means to keep the top k checkpoints according to the metric score; "latest_k" means to keep the last k checkpoints. The value of `k` is set via `ckpt_max_keep` |
| ckpt_max_keep | The maximum number of checkpoints to keep during training | 5 | Integer | \ |
| log_interval | The interval of printing logs (unit: epoch) | 100 | Integer | \ |
| val_while_train | Whether to enable the evaluation mode while training | True | True/False | If the value is True, please configure the eval data set synchronously |
| val_start_epoch | From which epoch to run the evaluation | 1 | Interger |  |
| val_interval   | Evaluation interval (unit: epoch)   | 1 | Interger |  |
| drop_overflow_update | Whether not updating network parameters when loss/gradient overflows | True | True/False | If value is true, network parameters will not be updated when overflow occurs |

## 2. Shared parameters (common)

Because the same parameter may need to be reused in different configuration sections, you can customize some common parameters in this section for easy management.

## 3. Model architecture (model)

In MindOCR, the network architecture of the model is divided into four modules: Transform, Backbone, Neck and Head. For details, please refer to [documentation](../../../mindocr/models/README.md), the following are the configuration instructions and examples of each module.

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| type | Network type | - | Currently supports rec/det; 'rec' means recognition task, 'det' means detection task |
| pretrained | Specify pre-trained weight path or url | null | Supports local checkpoint path or url |
| **transform:**| Transformation method configuration | null | |
| name | Specify transformation method name | - | Currently supports STN_ON |
| **backbone:** | Backbone network configuration ||
| name | Specify the backbone network class name or function name | - | Currently defined classes include rec_resnet34, rec_vgg7, SVTRNet and det_resnet18, det_resnet50, det_resnet152, det_mobilenet_v3. You can also customize new classes, please refer to the documentation for definition. |
| pretrained | Whether to load pre-trained backbone weights | False | Supports bool type or str type to be passed in. If it is True, the default weight will be downloaded and loaded through the url link defined in the backbone py file. If str is passed in, the local checkpoint path or url path can be specified for loading. |
| **neck:** | Network Neck configuration | |
| name | Neck class name | - | Currently defined classes include RNNEncoder, DBFPN, EASTFPN and PSEFPN. New classes can also be customized, please refer to the documentation for definition. |
| hidden_size | RNN hidden layer unit number | - | \ |
| **head:** | Network prediction header configuration ||
| name | Head class name | - | Currently supports CTCHead, AttentionHead, DBHead, EASTHead and PSEHead |
| weight_init | Set weight initialization | 'normal' | \ |
| bias_init | Set bias initialization | 'zeros' | \ |
| out_channels | Set the number of classes | - | \ |

> Note: For different networks, the configurable parameters of the backbone/neck/head module will be different. The specific configurable parameters are determined by the __init__ input parameter of the class specified by the `name` parameter of the module in the above table (For example, assume you specify the neck module is DBFPN. Since the DBFPN class initialization includes adaptive input parameters, parameters such as adaptive can be configured under the model.head in yaml.)

Reference example: [DBNet](../../../configs/det/dbnet/db_r50_mlt2017.yaml), [CRNN](../../../configs/rec/crnn/crnn_icdar15.yaml)

## 4. Postprocessing (postprocess)

Please see the code in [mindocr/postprocess](../../../mindocr/postprocess/)

| Parameter | Description | Example | Remarks |
| :---- | :---- | :---- | :---- |
| name | Post-processing class name | - | Currently supports DBPostprocess, EASTPostprocess, PSEPostprocess, RecCTCLabelDecode and RecAttnLabelDecode |
| character_dict_path | Recognition dictionary path | None | If None, then use the default dictionary [0-9a-z] |
| use_space_char | Set whether to add spaces to the dictionary | False | True/False |

> Note: For different post-processing methods (specified by name), the configurable parameters are different, and are determined by the input parameters of the initialization method `__init__` of the post-processing class.

Reference example: [DBNet](../../../configs/det/dbnet/db_r50_mlt2017.yaml), [PSENet](../../../configs/det/psenet/pse_r152_icdar15.yaml)


## 5. Evaluation metrics (metric)

Please see the code in [mindocr/metrics](../../../mindocr/metrics)

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| name | Metric class name | - | Currently supports RecMetric, DetMetric |
| main_indicator | Main indicator, used for comparison of optimal models | 'hmean' | 'acc' for recognition tasks, 'f-score' for detection tasks |
| character_dict_path | Recognition dictionary path | None | If None, then use the default dictionary "0123456789abcdefghijklmnopqrstuvwxyz" |
| ignore_space | Whether to filter spaces | True | True/False |
| print_flag | Whether to print log | False | If set True, then output information such as prediction results and standard answers |


## 6. Loss function (loss)

Please see the code in [mindocr/losses](../../../mindocr/losses)

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| name | loss function name | - | Currently supports L1BalancedCELoss, CTCLoss, AttentionLoss, PSEDiceLoss, EASTLoss and CrossEntropySmooth |
| pred_seq_len | length of predicted text | 26 | Determined by network architecture |
| max_label_len | The longest label length | 25 | The value is less than the length of the text predicted by the network |
| batch_size | single card batch size | 32 | \ |

> Note: For different loss functions (specified by name), the configurable parameters are different and determined by the input parameters of the selected loss function.

## 7. Learning rate adjustment strategy and optimizer (scheduler, optimizer, loss_scaler)

### Learning rate adjustment strategy (scheduler)

Please see the code in [mindocr/scheduler](../../../mindocr/scheduler)

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| scheduler | Learning rate scheduler name | 'constant' | Currently supports 'constant', 'cosine_decay', 'step_decay', 'exponential_decay', 'polynomial_decay', 'multi_step_decay' |
| min_lr | Minimum learning rate | 1e-6 | Lower lr bound for 'cosine_decay' schedulers. |
| lr | Learning rate value | 0.01 | |
| num_epochs | Number of total epochs | 200 | The number of total epochs for the entire training. |
| warmup_epochs | The number of epochs in the training learning rate warmp phase | 3 | For 'cosine_decay', 'warmup_epochs' indicates the epochs to warmup learning rate from 0 to `lr`. |
| decay_epochs | The number of epochs in the training learning rate decay phase | 10 | For 'cosine_decay' schedulers, decay LR to min_lr in `decay_epochs`. For 'step_decay' scheduler, decay LR by a factor of `decay_rate` every `decay_epochs`. |


### optimizer

Please see the code location: [mindocr/optim](../../../mindocr/optim)

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| opt | Optimizer name | 'adam' | Currently supports 'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'lion', 'nadam', 'adan', 'rmsprop', 'adagrad', 'lamb'. |
| filter_bias_and_bn | Set whether to exclude the weight decrement of bias and batch norm | True | If True, weight decay will not apply on BN parameters and bias in Conv or Dense layers. |
| momentum | momentum | 0.9 | \ |
| weight_decay | weight decay rate | 0 | It should be noted that weight decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay value of current step. |
| nesterov | Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. | False | True/False |


### Loss scaling (loss_scaler)

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| type | Loss scaling method type | static | Currently supports static, dynamic |
| loss_scale | Loss scaling value | 1.0 | \ |
| scale_factor | When using dynamic loss scaler, the coefficient to dynamically adjust the loss_scale | 2.0 | At each training step, the loss scaling value is updated to `loss_scale`/`scale_factor` when overflow occurs. |
| scale_window | When using the dynamic loss scaler, when there is no overflow after the scale_window training step, enlarge the loss_scale by scale_factor times | 1000 | If the continuous `scale_window` steps does not overflow, the loss will be increased by `loss_scale` * `scale_factor` to update the scaling number |


## 8. Training, evaluation and predict process (train, eval, predict)

The configuration of the training process is placed under `train`, and the configuration of the evaluation phase is placed under `eval`. Note that during model training, if the training-while-evaluation mode is turned on, that is, when val_while_train=True, an evaluation will be run according to the configuration under `eval` after each epoch is trained. During the non-training phase, only the `eval` configuration is read when only running model evaluation.

### Training process (train)

| Parameter | Description | Default | Remarks |
| :---- | :---- | :---- | :---- |
| ckpt_save_dir | Set model save path | ./tmp_rec | \ |
| resume | Resume training after training is interrupted, you can set True/False, or specify the ckpt path that needs to be loaded to resume training | False | If True, load resume_train.ckpt under the ckpt_save_dir directory to continue training. You can also specify the ckpt file path to load and resume training. |
| dataset_sink_mode | Whether the data is directly sinked to the processor for processing | - | If set to True, the data sinks to the processor, and the data can be returned at least after the end of each epoch |
| gradient_accumulation_steps | Number of steps to accumulate the gradients | 1 | Each step represents a forward calculation, and a reverse correction is performed after the gradient accumulation is completed. |
| clip_grad | Whether to clip the gradient | False | If set to True, gradients are clipped to `clip_norm` |
| clip_norm | The norm of clipping gradient if set clip_grad as True | 1 | \ |
| ema | Whether to use EMA algorithm | False | \ |
| ema_decay | EMA decay rate | 0.9999 | \ |
| pred_cast_fp32 | Whether to cast the data type of logits to fp32 | False | \ |
| **dataset** | Dataset configuration | | For details, please refer to [Data document](../../../mindocr/data/README.md) |
| type | Dataset class name | - | Currently supports LMDBDataset, RecDataset and DetDataset |
| dataset_root | The root directory of the dataset | None | Optional |
| data_dir | The subdirectory where the dataset is located | - | If `dataset_root` is not set, please set this to the full directory |
| label_file | The label file path of the dataset | - | If `dataset_root` is not set, please set this to the full path, otherwise just set the subpath |
| sample_ratio | Data set sampling ratio | 1.0 | If value < 1.0, random selection |
| shuffle | Whether to shuffle the data order | True if undering training, otherwise False | True/False |
| transform_pipeline | Data processing flow | None | For details, please see [transforms](../../../mindocr/data/transforms/README.md) |
| output_columns | Data loader (data loader) needs to output a list of data attribute names (given to the network/loss calculation/post-processing) (type: list), and the candidate data attribute names are determined by transform_pipeline. | None | If the value is None, all columns are output. Take crnn as an example, output_columns: \['image', 'text_seq'\] |
| net_input_column_index | In output_columns, the indices of the input items required by the network construct function | [0] | \ |
| label_column_index | In output_columns, the indices of the input items required by the loss function | [1] | \ |
| **loader** | Data Loading Settings ||
| shuffle | Whether to shuffle the data order for each epoch | True if undering training, otherwise False | True/False |
| batch_size | Batch size of a single card | - | \ |
| drop_remainder | Whether to drop the last batch of data when the total data cannot be divided by batch_size | True if undering training, otherwise False | \ |
| max_rowsize | Specifies the maximum space allocated by shared memory when copying data between multiple processes | 64 | Default value: 64 |
| num_workers | Specifies the number of concurrent processes/threads for batch operations | n_cpus / n_devices - 2 | This value should be greater than or equal to 2 |

Reference example: [DBNet](../../../configs/det/dbnet/db_r50_mlt2017.yaml), [CRNN](../../../configs/rec/crnn/crnn_icdar15.yaml)

### Evaluation process (eval)

The parameters of `eval` are basically the same as `train`, only a few additional parameters are added, and for the rest, please refer to the parameter description of `train` above.

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| ckpt_load_path | Set model loading path | - | \ |
| num_columns_of_labels | Set the number of labels in the dataset output columns | None | If None, assuming the columns after image (data[1:]) are labels. If not None, the num_columns_of_labels columns after image (data[1:1+num_columns_of_labels]) are labels, and the remaining columns are additional info like image_path. |
| drop_remainder | Whether to discard the last batch of data when the total number of data cannot be divided by batch_size | True if undering training, otherwise False | It is recommended to set it to False when doing model evaluation. If it cannot be divisible, mindocr will automatically select a batch size that is the largest divisible |
