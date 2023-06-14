# Configuration parameter description

- [system](#1-training-environment-context-parameters-system)
- [common](#2-common-parameters-common)
- [model](#3-model-architecture-model)
- [postprocess](#4-postprocessing-postprocess)
- [metric](#5-evaluation-metrics-metric)
- [loss](#6-loss-function-loss)
- [scheduler, optimizer, loss_scaler](#7-learning-rate-adjustment-strategy-and-optimizer-scheduler-optimizer-loss_scaler)
  - [scheduler](#learning-rate-adjustment-strategy-scheduler)
  - [optimizer](#optimizer)
  - [loss_scaler](#loss-scaling-loss_scaler)
- [train, eval](#8-training-and-evaluation-process-train-eval)
  - [train](#training-process-train)
  - [eval](#evaluation-process-eval)

This document takes `configs/rec/crnn/crnn_icdar15.yaml` as an example to describe the usage of parameters in detail.

## 1. Training environment context parameters (system)

| Parameter | Usage | Default | Optional Values ​​| Remarks |
| ---- | ---- | ---- | ---- | ---- |
| mode | Set the running and compiling mode | 0 | 0 / 1 | 0: means running in GRAPH_MODE mode; 1: PYNATIVE_MODE mode |
| distribute | Set whether to enable parallel training | True | True / False | \ |
| amp_level | Set mixed precision mode | O0 | O0/O1/O2/O3 | 'O0' - no change. <br> 'O1' - convert the cells and operations in the whitelist to float16 precision, and keep the rest in float32 precision. <br> 'O2' - Keep the cells and operations in the blacklist with float32 precision, and convert the rest to float16 precision. <br> 'O3' - Convert all networks to float16 precision. |
| seed | Random seed | 42 | Integer | \ |
| ckpt_save_policy | The policy for checkpoint saving | top_k | "top_k" or "latest_k" | "top_k" means to keep the top k checkpoints according to the metric score; "latest_k" means to keep the last k checkpoints. The value of `k` is set via `ckpt_max_keep` |
| ckpt_max_keep | The maximum number of checkpoints to keep during training | 5 | Integer | \ |
| log_interval | The interval of printing logs | 100 | Integer | \ |
| val_while_train | Whether to enable the evaluation mode while training | True | True/False | If the value is True, please configure the eval data set synchronously |
| drop_overflow_update | Whether to execute the optimizer when overflow occurs | True | True/False | If the value is True, the optimizer will not be executed when overflow occurs |

## 2. Common parameters (common)

Because the same parameter may need to be reused in different configuration sections, you can customize some common parameters in this section for easy management.

## 3. Model architecture (model)

In MindOCR, the network architecture of the model is divided into four parts: Transform, Backbone, Neck and Head. For details, please refer to [documentation](../mindocr/models/README.md), the following are the configuration instructions and examples of each part:

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| type | Network type | - | Currently supports rec/det; 'rec' means recognition task, 'det' means detection task |
| pretrained | Initialize the model with pretrained model weights | null | Supports local checkpoint path or url |
| **transform**| Configure transformation method | null | |
| name | Transformation method name | - | Currently supports STN_ON |
| **backbone** | Configure backbone network ||
| name | Backbone network class name | - | Currently defined classes include rec_resnet34, rec_vgg7, SVTRNet and det_resnet18, det_resnet50, det_resnet152, det_mobilenet_v3. You can also customize new classes, please refer to the documentation for definition. |
| pretrained | Whether to use the pre-trained backbone | False | Support local checkpoint path or url |
| **neck** | Configure Network Neck | |
| name | Neck class name | - | Currently defined classes include RNNEncoder, DBFPN, EASTFPN and PSEFPN. New classes can also be customized, please refer to the documentation for definition. |
| hidden_size | RNN hidden layer unit number | - | \ |
| **head** | Set network prediction header ||
| name | Head class name | - | Currently supports CTCHead, AttentionHead, DBHead, EASTHead and PSEHead |
| weight_init | Set weight initialization | 'normal' | \ |
| bias_init | Set bias initialization | 'zeros' | \ |
| out_channels | Set the number of classes | - | \ |

## 4. Postprocessing (postprocess)

Please see the code in [mindocr/postprocess](../mindocr/postprocess/)

| Parameter | Usage | Example | Remarks |
| :---- | :---- | :---- | :---- |
| name | Post-processing class name | - | Currently supports DBPostprocess, EASTPostprocess, PSEPostprocess, RecCTCLabelDecode and RecAttnLabelDecode |
| character_dict_path | Recognition dictionary path | None | If None, then use the default dictionary [0-9a-z] |
| use_space_char | Set whether to add spaces to the dictionary | False | True/False |


## 5. Evaluation metrics (metric)

Please see the code in [mindocr/metrics](../mindocr/metrics)

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| name | Metric class name | - | Currently supports RecMetric, DetMetric |
| main_indicator | Main indicator, used for comparison of optimal models | 'hmean' | 'acc' for recognition tasks, 'f-score' for detection tasks |
| character_dict_path | Recognition dictionary path | None | If None, then use the default dictionary [0-9a-z] |
| ignore_space | Whether to filter spaces | True | True/False |
| print_flag | Whether to print log | False | If set True, then output information such as prediction results and standard answers |


## 6. Loss function (loss)

Please see the code in [mindocr/losses](../mindocr/losses)

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| name | loss function name | - | Currently supports L1BalancedCELoss, CTCLoss, AttentionLoss, PSEDiceLoss, EASTLoss and CrossEntropySmooth |
| pred_seq_len | length of predicted text | 26 | Determined by network architecture |
| max_label_len | The longest label length | 25 | The value is less than the length of the text predicted by the network |
| batch_size | single card batch size | 32 | \ |


## 7. Learning rate adjustment strategy and optimizer (scheduler, optimizer, loss_scaler)

### Learning rate adjustment strategy (scheduler)

Please see the code in [mindocr/scheduler](../mindocr/scheduler)

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| scheduler | Learning rate adjustment function name | 'constant' | Currently supports 'constant', 'cosine_decay', 'step_decay', 'exponential_decay', 'polynomial_decay', 'multi_step_decay' |
| min_lr | Minimum learning rate | 1e-6 | Lower lr bound for 'cosine_decay' schedulers. |
| lr | Learning rate value | 0.01 | Upper lr bound for scheduler |
| num_epochs | Number of total epochs | 200 | The number of total epochs for the entire training. |
| warmup_epochs | Epochs to warmup LR, if scheduler supports | 3 | For 'cosine_decay', 'warmup_epochs' indicates the epochs to warmup learning rate from `min_lr` to `lr`. |
| decay_epochs | Learning rate decrement epoch number | 10 | For 'cosine_decay' schedulers, decay LR to min_lr in `decay_epochs`. For 'step_decay' scheduler, decay LR by a factor of `decay_rate` every `decay_epochs`. |


### optimizer

Please see the code location: [mindocr/optim](../mindocr/optim)

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| opt | Optimizer name | 'adam' | Currently supports 'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'lion', 'nadam', 'adan', 'rmsprop', 'adagrad', 'lamb'. |
| filter_bias_and_bn | Set whether to exclude the weight decrement of bias and batch norm | True | If True, weight decay will not apply on BN parameters and bias in Conv or Dense layers. |
| momentum | momentum | 0.9 | \ |
| weight_decay | weight decay rate | 0 | It should be noted that weight decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay value of current step. |
| nesterov | Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. | False | True/False |


### Loss scaling (loss_scaler)

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| type | Loss scaling management type | static | Currently supports static, dynamic |
| loss_scale | Initial loss scaling value | 1.0 | \ |
| scale_factor | Coefficient for updating the loss_scale if using dynamic scaler | 2.0 | At each training step, the loss scaling value is updated to `loss_scale`/`scale_factor` when overflow occurs. |
| scale_window | Maximum continuous training steps that do not have overflow to update the loss scale if using dynamic scaler | 1000 | If the continuous `scale_window` steps does not overflow, the loss will be increased by `loss_scale` * `scale_factor` to reduce the scaling number |


## 8. Training, evaluation and predict process (train, eval, predict)

The configuration of the training process is placed under `train`, and the configuration of the evaluation phase is placed under `eval`. Note that during model training, if the training-while-evaluation mode is turned on, that is, when val_while_train=True, an evaluation will be run according to the configuration under `eval` after each epoch is trained. During the non-training phase, only the `eval` configuration is read when only running model evaluation.

### Training process (train)

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| ckpt_save_dir | Set model save path | ./tmp_rec | \ |
| dataset_sink_mode | Whether the data is directly sinked to the processor for processing | - | If set to True, the data sinks to the processor, and the data can be returned at least after the end of each epoch |
| gradient_accumulation_steps | Number of steps to accumulate the gradients | 1 | Each step represents a forward calculation, and a reverse correction is performed after the gradient accumulation is completed. |
| clip_grad | Whether to clip the gradient | False | If set to True, gradients are clipped to `clip_norm` |
| clip_norm | The norm of clipping gradient if set clip_grad as True | 1 | \ |
| ema | Whether to use EMA weights | False | \ |
| ema_decay | EMA decay rate | 0.9999 | \ |
| pred_cast_fp32 | Whether to cast the data type of logits to fp32 | False | \ |
| **dataset** | Dataset configuration | For details, please refer to [Data document](../mindocr/data/README.md) ||
| type | Dataset class name | - | Currently supports LMDBDataset, RecDataset and DetDataset |
| dataset_root | The root directory of the dataset | None | Optional |
| data_dir | The subdirectory where the dataset is located | - | If `dataset_root` is not set, please set this to the full directory |
| label_file | The label file path of the dataset | - | If `dataset_root` is not set, please set this to the full path, otherwise just set the subpath |
| sample_ratio | Data set sampling ratio | 1.0 | If value < 1.0, random selection |
| shuffle | Whether to shuffle the data order | True if undering training, otherwise False | True/False |
| transform_pipeline | Data processing flow | None | For details, please see [transforms](../mindocr/data/transforms) |
| output_columns | Each data feature name | None | If None, then output all columns |
| net_input_column_index | In output_columns, the indices of the input items required by the network construct function | [0] | \ |
| label_column_index | In output_columns, the indices of the input items required by the loss function | [1] | \ |
| **loader** | Data Loading Settings ||
| shuffle | Whether to shuffle the data order for each epoch | True if undering training, otherwise False | True/False |
| batch_size | Batch size of a single card | - | \ |
| drop_remainder | Whether to drop the last batch of data when the total data cannot be divided by batch_size | True if undering training, otherwise False | \ |
| max_rowsize | Specifies the maximum space allocated by shared memory when copying data between multiple processes | 64 | Default value: 64 |
| num_workers | Specifies the number of concurrent processes/threads for batch operations | n_cpus / n_devices - 2 | This value should be greater than or equal to 2 |


### Evaluation process (eval)

The parameters of `eval` are basically the same as `train`, only a few additional parameters are added, and for the rest, please refer to the parameter description of `train` above.

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| ckpt_load_path | Set model loading path | - | \ |
| num_columns_of_labels | Set the number of labels in the dataset output columns | None | If None, assuming the columns after image (data[1:]) are labels. If not None, the num_columns_of_labels columns after image (data[1:1+num_columns_of_labels]) are labels, and the remaining columns are additional info like image_path. |
| drop_remainder | Whether to discard the last batch of data when the total number of data cannot be divided by batch_size | True if undering training, otherwise False | It is recommended to set it to False when doing model evaluation. If it cannot be divisible, mindocr will automatically select a batch size that is the largest divisible |


### Predict process (predict)

The parameters of `predict` are basically the same as `train`, only a few additional parameters are added, and for the rest, please refer to the parameter description of `train` above.

| Parameter | Usage | Default | Remarks |
| :---- | :---- | :---- | :---- |
| ckpt_load_path | Set model loading path | - | \ |
| vis_font_path | Font loading path for visual display of inference results | tools/utils/simfang.ttf | \ |
