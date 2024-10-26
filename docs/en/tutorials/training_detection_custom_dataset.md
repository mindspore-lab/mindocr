# Training Detection Network with Custom Datasets

This document provides tutorials on how to train text detection networks using custom datasets.


- [Training Detection Network with Custom Datasets](#training-detection-network-with-custom-datasets)
  - [1. Dataset preperation](#1-dataset-preperation)
    - [1.1 Preparing Training Data](#11-preparing-training-data)
    - [1.2 Preparing Validation Data](#12-preparing-validation-data)
  - [2. Configuration File Preperation](#2-configuration-file-preperation)
    - [2.1 Configure train/validation datasets](#21-configure-trainvalidation-datasets)
    - [2.2 Configure train/validation transform pipelines](#22-configure-trainvalidation-transform-pipelines)
    - [2.3 Configure the model architecture](#23-configure-the-model-architecture)
    - [2.4 Configure training hyperparameters](#24-configure-training-hyperparameters)
  - [3. Model Training, Evaluation, and Inference](#3-model-training-evaluation-and-inference)
    - [3.1 Training](#31-training)
    - [3.2 Evaluation](#32-evaluation)
    - [3.3 Inference](#33-inference)
      - [3.3.1 Environment Preparation](#331-environment-preparation)
      - [3.3.2 Model Conversion](#332-model-conversion)
      - [3.3.3 Inference (Python)](#333-inference-python)

## 1. Dataset preperation

Currently, MindOCR detection network supports two input formats, namely
- `Common Dataset`：A file format that stores images, text bounding boxes, and transcriptions. An example of the target file format is:
``` text
img_1.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```

It is read by [DetDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py). If your dataset is not in the same format as the example format, see [instructions](../datasets/converters.md) on how convert different datasets' annotations into the supported format.

- `SynthTextDataset`: A file format provided by [SynthText800k](https://github.com/ankush-me/SynthText). More details about this dataset can be found [here](../datasets/synthtext.md). The annotation file is a `.mat` file consisting of `imnames`(image names), `wordBB`(word-level bounding-boxes), `charBB`(character-level bounding boxes), and `txt` (text strings). It is read by [SynthTextDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py). Users can take `SynthTextDataset` as a reference to write their custom dataset class.

We recommend users to prepare text detection datasets in the `Common Dataset` format, and then use [DetDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/det_dataset.py) to load the data. The following tutorials further explain on the detailed steps.


### 1.1 Preparing Training Data
Please place all training images in a single folder, and specify a txt file `train_det.txt` at a higher directory to label all training image names and corresponding labels. An example of the txt file is as follows:

```
# File Name	# A list of dictionaries
img_1.jpg\t[{"transcription": "Genaxis Theatre", "points": [[377, 117], [463, 117], [465, 130], [378, 130]]}, {"transcription": "[06]", "points": [[493, 115], [519, 115], [519, 131], [493, 131]]}, {...}]
img_2.jpg\t[{"transcription": "guardian", "points": [[642, 250], [769, 230], [775, 255], [648, 275]]}]
...
```
*Note*: Please separate image names and labels using \tab, and avoid using spaces or other delimiters.

The final training set will be stored in the following format:

```
|-data
    |- train_det.txt
    |- training
        |- img_1.jpg
        |- img_2.jpg
        |- img_3.jpg
        | ...
```

### 1.2 Preparing Validation Data
Similarly, please place all validation images in a single folder, and specify a txt file `val_det.txt` at a higher directory to label all validation image names and corresponding labels. The final validation set will be stored in the following format:

```
|-data
    |- val_det.txt
    |- validation
        |- img_1.jpg
        |- img_2.jpg
        |- img_3.jpg
        | ...
```

## 2. Configuration File Preperation

To prepare the corresponding configuration file, users should specify the directories for the training and validation datasets.

### 2.1 Configure train/validation datasets

Please select `configs/det/dbnet/db_r50_icdar15.yaml` as the initial configuration file and modify the `train.dataset` and `eval.dataset` fields in it.

```yaml
...
train:
  ...
  dataset:
    type: DetDataset                                                  # File reading method. Here we use the `Common Dataset` format
    dataset_root: dir/to/data/                                        # Root directory of the data
    data_dir: training/                                               # Training dataset directory. It will be concatenated with `dataset_root` to form a complete path.
    label_file: train_det.txt                                       # Path of the training label. It will be concatenated with `dataset_root` to form a complete path.
...
eval:
  dataset:
    type: DetDataset                                                  # File reading method. Here we use the `Common Dataset` format
    dataset_root: dir/to/data/                                        # Root directory of the data
    data_dir: validation/                                             # Validation dataset directory. It will be concatenated with `dataset_root` to form a complete path.
    label_file: val_det.txt                                     # Path of the validation label. It will be concatenated with `dataset_root` to form a complete path.
  ...
```

### 2.2 Configure train/validation transform pipelines

Take the `train.dataset.transform_pipeline` field in the `configs/det/dbnet/dbnet_r50_icdar15.yaml` as an example. It specifies a set of transformations applied on the image or labels to generate the data as the model inputs or the loss function inputs. These transform functions are defined in `mindocr/data/transforms`.

```yaml
...
train:
...
  dataset:
    transform_pipeline:
      - DecodeImage:
          img_mode: RGB
          to_float32: False
      - DetLabelEncode:
      - RandomColorAdjust:
          brightness: 0.1255  # 32.0 / 255
          saturation: 0.5
      - RandomHorizontalFlip:
          p: 0.5
      - RandomRotate:
          degrees: [ -10, 10 ]
          expand_canvas: False
          p: 1.0
      - RandomScale:
          scale_range: [ 0.5, 3.0 ]
          p: 1.0
      - RandomCropWithBBox:
          max_tries: 10
          min_crop_ratio: 0.1
          crop_size: [ 640, 640 ]
          p: 1.0
      - ValidatePolygons:
      - ShrinkBinaryMap:
          min_text_size: 8
          shrink_ratio: 0.4
      - BorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - NormalizeImage:
          bgr_to_rgb: False
          is_hwc: True
          mean: imagenet
          std: imagenet
      - ToCHWImage:
  ...
```

- `DecodeImage` and `DetLabelEncode`: the two transform functions parse the strings in `train_det.txt` file, load both the image and the labels, and save them as a dictionary;

- `RandomColorAdjust`,  `RandomHorizontalFlip`, `RandomRotate`, `RandomScale`, and `RandomCropWithBBox`: these transform functions perform typical image augmentation operations. Except for `RandomColorAdjust`, all other functions alter the bounding box labels;

- `ValidatePolygons`: it filters out the bounding boxes that are outside of the image due to previous augmentations;

- `ShrinkBinaryMap` and `BorderMap`: they make the binary map and the border map needed for dbnet training;

- `NormalizeImage`: it normalizes the image by the mean and variance of the ImageNet dataset;

- `ToCHWImage`: it changes `HWC` images to `CHW` images.

For validation transform pipeline, all image augmentation operations are removed, and replaced by a simple resize function:

```yaml
eval:
  dataset
    transform_pipeline:
      - DecodeImage:
          img_mode: RGB
          to_float32: False
      - DetLabelEncode:
      - DetResize:
          target_size: [ 736, 1280 ]
          keep_ratio: False
          force_divisable: True
      - NormalizeImage:
          bgr_to_rgb: False
          is_hwc: True
          mean: imagenet
          std: imagenet
      - ToCHWImage:
```
More tutorials on transform functions can be found in the [transform tutorial](transform_tutorial.md).

### 2.3 Configure the model architecture

Although different models have different architectures, MindOCR formulates them as a general three-stage architecture: `[backbone]->[neck]->[head]`. Take `configs/det/dbnet/dbnet_r50_icdar15.yaml` as an example:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet50  # Only ResNet50 is supported at the moment
    pretrained: True    # Whether to use weights pretrained on ImageNet
  neck:
    name: DBFPN         # FPN part of the DBNet
    out_channels: 256
    bias: False
    use_asf: False      # Adaptive Scale Fusion module from DBNet++ (use it for DBNet++ only)
  head:
    name: DBHead
    k: 50               # amplifying factor for Differentiable Binarization
    bias: False
    adaptive: True      # True for training, False for inference
```
 The backbone, neck, and head modules are all defined under `mindocr/models/backbones`, `mindocr/models/necks`, and `mindocr/models/heads`.

### 2.4 Configure training hyperparameters

Some training hyperparameters in `configs/det/dbnet/dbnet_r50_icdar15.yaml` are defined as follows:
```yaml
metric:
  name: DetMetric
  main_indicator: f-score

loss:
  name: DBLoss
  eps: 1.0e-6
  l1_scale: 10
  bce_scale: 5
  bce_replace: bceloss

scheduler:
  scheduler: polynomial_decay
  lr: 0.007
  num_epochs: 1200
  decay_rate: 0.9
  warmup_epochs: 3

optimizer:
  opt: SGD
  filter_bias_and_bn: false
  momentum: 0.9
  weight_decay: 1.0e-4
```
It uses `SGD` optimizer (in `mindocr/optim/optim.factory.py`) and `polynomial_decay` (in `mindocr/scheduler/scheduler_factory.py`) as the learning scheduler. The loss function is `DBLoss` (in `mindocr/losses/det_loss.py`) and the evaluation metric is `DetMetric` ( in `mindocr/metrics/det_metrics.py`).


## 3. Model Training, Evaluation, and Inference

When all configurations have been specified, users can start training their models. MindOCR supports evaluation and inference after the model is trained.

### 3.1 Training

* Standalone training

In standalone training, the model is trained on a single device (`device:0` by default). Users should set `system.distribute` in yaml config file to be `False`, and the `system.device_id` to the target device id if users want to run this model on a device other than `device:0`.

Take `configs/det/dbnet/db_r50_icdar15.yaml` as an example, the training command is:

```shell
python tools/train.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

* Distributed training

In distributed training, `distribute` in yaml config file should be True. On both GPU and Ascend devices, users can use `mpirun` to launch distributed training. For example, using `device:0` and `device:1` to train:

```shell
# n is the number of GPUs/NPUs
mpirun --allow-run-as-root -n 2 python tools/train.py --config configs/det/dbnet/db_r50_icdar15.yaml
```

Sometimes, users may want to specify the device ids to run distributed training, for example, `device:2` and `device:3`.


 On GPU devices, before running the `mpirun` command above, users can run the following command:

```shell
export CUDA_VISIBLE_DEVICES=2,3
```

On Ascend devices, users should create a `rank_table.json` like this:
```json
Copy{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}

```
To get the `device_ip` of the target device, run `cat /etc/hccn.conf` and look for the value of `address_x`, which is the ip address. More details can be found in [distributed training tutorial](distribute_train.md).

### 3.2 Evaluation

To evaluate the accuracy of the trained model, users can use `tools/eval.py`.

Take standalone evaluation as an example. In the yaml config file, `system.distribute` should be `False`; the `eval.ckpt_load_path` should be the target ckpt path; `eval.dataset_root`, `eval.data_dir`, and `eval.label_file` should be correctly specified. Then the evaluation can be started by running:

```Shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml
```

MindOCR also supports to specify the arguments in the command line, by running:
```Shell
python tools/eval.py -c=configs/det/dbnet/db_r50_icdar15.yaml \
            --opt eval.ckpt_load_path="/path/to/local_ckpt.ckpt" \
                  eval.dataset_root="/path/to/val_set/root" \
                  eval.data_dir="val_set/dir"\
                  eval.label_file="val_set/label"

```

### 3.3 Inference

MindOCR inference supports Ascend310/Ascend310P devices, supports [MindSpore Lite](https://www.mindspore.cn/lite) and
[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)
inference backend. [Inference Tutorial](../inference/inference_tutorial.md) gives detailed steps on how to run inference with MindOCR, which include mainly three steps: environment preparation, model conversion, and inference.

#### 3.3.1 Environment Preparation

Please refer to the [environment installation](../inference/environment.md) for more information, and pay attention to selecting the ACL/Lite environment based on the model.

#### 3.3.2 Model Conversion

Before runing infernence, users need to export a MindIR file from the trained checkpoint. [MindSpore IR (MindIR)](https://www.mindspore.cn/docs/en/r2.0/design/mindir.html) is a function-style IR based on graph representation. The MindIR filew stores the model structure and weight parameters needed for inference.

Given the trained dbnet checkpoint file, user can use the following commands to export MindIR:

```shell
python tools/export.py --model_name_or_config dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name_or_config configs/det/dbnet/db_r50_icdar15.yaml --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt.ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. It may change when the model is changed.

Please refer to the [Conversion Tutorial](../inference/convert_tutorial.md) for more details about model conversion.

#### 3.3.3 Inference (Python)


 After model conversion, the `output.mindir` is obtained. Users can go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --device=Ascend \
    --device_id=0 \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/dbnet/db_r50_icdar15.yaml \
    --backend=lite \
    --res_save_dir=results_dir
```

Please refer to the [Inference Tutorials](../inference/inference_tutorial.md) chapter `4.1 Command example` on more examples of inference commands.
