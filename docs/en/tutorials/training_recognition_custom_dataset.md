# Training Recognition Network with Custom Datasets

This document provides tutorials on how to train recognition networks using custom datasets, including the training of recognition networks in Chinese and English languages.

## Dataset preperation

Currently, MindOCR recognition network supports two input formats, namely
- `Common Dataset`：A file format that stores images and text files. It is read by [RecDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/rec_dataset.py).
- `LMDB Dataset`: A file format provided by [LMDB](https://www.symas.com/lmdb). It is read by [LMDBDataset](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/rec_lmdb_dataset.py).

The following tutorials take the use of the `Common Dataset` file format as an example.


### Preparing Training Data
Please place all training images in a single folder, and specify a txt file at a higher directory to label all training image names and corresponding labels. An example of the txt file is as follows:

```
# File Name	# Corresponding label
word_421.png	菜肴
word_1657.png	你好
word_1814.png	cathay
```
*Note*: Please separate image names and labels using \tab, and avoid using spaces or other delimiters.

The final training set will be stored in the following format:

```
|-data
    |- gt_training.txt
    |- training
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

### Preparing Validation Data
Similarly, please place all validation images in a single folder, and specify a txt file at a higher directory to label all validation image names and corresponding labels. The final validation set will be stored in the following format:

```
|-data
    |- gt_validation.txt
    |- validation
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

## Dictionary Preperation

To train recognition networks for different languages, users need to configure corresponding dictionaries. Only characters that exist in the dictionary will be correctly predicted by the model. MindOCR currently provides three dictionaries, corresponding to Default, Chinese and English respectively.
- `Default Dictionary`：includes lowercase English letters and numbers only. If users do not configure the dictionay, this one will be used by default.
- `English Dictionary`：includes uppercase and lowercase English letters, numbers and punctuation marks, it is place at `mindocr/utils/dict/en_dict.txt`.
- `Chinese Dictionary`：includes commonly used Chinese characters, uppercase and lowercase English letters, numbers, and punctuation marks, it is placed at `mindocr/utils/dict/ch_dict.txt`.

Currently, MindOCR does not provide a dictionary configuration for other languages. This feature will be released in a upcoming version.

## Configuration File Preperation

To configure the corresponding configuration file for a specific network architecture, users need to provide the necessary settings. As an example, we can take CRNN (with backbone Resnet34) as an example.

### Configure an English Model

Please select `configs/rec/crnn/crnn_resnet34.yaml` as the initial configuration file and modify the `train.dataset` and `eval.dataset` fields in it.

```yaml
...
train:
  ...
  dataset:
    type: RecDataset                                                  # File reading method. Here we use the `Common Dataset` format
    dataset_root: dir/to/data/                                        # Root directory of the data
    data_dir: training/                                               # Training dataset directory. It will be concatenated with `dataset_root` to form a complete path.
    label_file: gt_training.txt                                       # Path of the training label. It will be concatenated with `dataset_root` to form a complete path.
...
eval:
  dataset:
    type: RecDataset                                                  # File reading method. Here we use the `Common Dataset` format
    dataset_root: dir/to/data/                                        # Root directory of the data
    data_dir: validation/                                             # Validation dataset directory. It will be concatenated with `dataset_root` to form a complete path.
    label_file: gt_validation.txt                                     # Path of the validation label. It will be concatenated with `dataset_root` to form a complete path.
  ...
```

And also modify the corresponding dictionary location to the the English dictionary path.

```yaml
...
common:
  character_dict_path: &character_dict_path mindocr/utils/dict/en_dict.txt
...
```

To use the complete English dictionary, users need to modify the `common:num_classes` attribute in the corresponding configuration file, as the initial configuration file’s dictionary only includes lowercase English and numbers.

```yaml
...
common:
  num_classes: &num_classes 95                                        # The number is equal to the number of dictionary characters plus 1
...
```

If the network needs to output spaces, it is necessary to modify the `common.use_space_char` attribute and the `common: num_classes` attribute as follows:

```bash
...
common:
  num_classes: &num_classes 96                                      # The number must be equal to the number of characters in the dictionary plus the number of spaces plus 1.
  use_space_char: &use_space_char True                                # Output `space` character additonaly
...
```


##### Configuring a custom English dictionary

The user can add, delete, or modify characters within the dictionary as needed. It is important to note that characters must be separated by newline characters `\n`, and it is necessary to avoid having duplicate characters in the same dictionary. Additionally, the user must also modify the `common: num_classes` attribute in the configuration file to ensure that it is equal to the number of characters in the dictionary plus 1 (in the case of a seq2seq model, it is equal to the number of characters in the dictionary plus 2).

### Configure an Chinese Model

Please select `configs/rec/crnn/crnn_resnet34_ch.yaml` as the initial configuration file and modify the `train.dataset` and `eval.dataset` fields in it.

```yaml
...
train:
  ...
  dataset:
    type: RecDataset                                                  # File reading method. Here we use the `Common Dataset` format
    dataset_root: dir/to/data/                                        # Root directory of the data
    data_dir: training/                                               # Training dataset directory. It will be concatenated with `dataset_root` to form a complete path.
    label_file: gt_training.txt                                       # Path of the training label. It will be concatenated with `dataset_root` to form a complete path.
...
eval:
  dataset:
    type: RecDataset                                                  # File reading method. Here we use the `Common Dataset` format
    dataset_root: dir/to/data/                                        # Root directory of the data
    data_dir: validation/                                             # Validation dataset directory. It will be concatenated with `dataset_root` to form a complete path.
    label_file: gt_validation.txt                                     # Path of the validation label. It will be concatenated with `dataset_root` to form a complete path.
  ...
```

And also modify the corresponding dictionary location to the the Chinese dictionary path.

```yaml
...
common:
  character_dict_path: &character_dict_path mindocr/utils/dict/ch_dict.txt
...
```

If the network needs to output spaces, it is necessary to modify the `common.use_space_char` attribute and the `common: num_classes` attribute as follows:

```yaml
...
common:
  num_classes: &num_classes 6625                                      # The number must be equal to the number of characters in the dictionary plus the number of spaces plus 1.
  use_space_char: &use_space_char True                                # Output `space` character additonaly
...
```

##### Configuring a custom Chinese dictionary

The user can add, delete, or modify characters within the dictionary as needed. It is important to note that characters must be separated by newline characters `\n`, and it is necessary to avoid having duplicate characters in the same dictionary. Additionally, the user must also modify the `common: num_classes` attribute in the configuration file to ensure that it is equal to the number of characters in the dictionary plus 1 (in the case of a seq2seq model, it is equal to the number of characters in the dictionary plus 2).

## Model Training

When all datasets and configuration files are ready, users can start training models with their own data. As each model has different training methods, users can refer to the corresponding model introduction documentation for the **Model Training** and **Model Evaluation** sections. Here, we will only use CRNN as an example.


### Preparing Pre-trained Model

Users can use the pre-trained models that we provide as a starting point for training. Pre-trained models can often improve the convergence speed and even accuracy of the model. Taking the Chinese model as an example, the url for the pre-trained model that we provide is <https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt>. Users only need to add `model.pretrained` with the corresponding url in the configuration file as follows:

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet34
    pretrained: False
  neck:
    name: RNNEncoder
    hidden_size: 64
  head:
    name: CTCHead
    out_channels: *num_classes
  pretrained: https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c.ckpt
...
```

If users encounter network issues, they can try downloading the pre-trained model to their local machine in advance, and then change `model.pretrained` to the local path as follows:

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet34
    pretrained: False
  neck:
    name: RNNEncoder
    hidden_size: 64
  head:
    name: CTCHead
    out_channels: *num_classes
  pretrained: /local_path_to_the_ckpt/crnn_resnet34_ch-7a342e3c.ckpt
...
```

If users do not need to use the pre-trained model, they can simply delete `model.pretrained`.

### Start Training

#### Distributed Training

In the case of a large amount of data, we recommend that users use distributed training. For distributed training across multiple Ascend 910 devices or GPU devices, please modify the configuration parameter `system.distribute` to True, for example:

```shell
# To perform distributed training on 4 GPU/Ascend devices
mpirun -n 4 python tools/train.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

#### Single Device Training

If you want to train or fine-tune the model on a smaller dataset without distributed training, please modify the configuration parameter `system.distribute` to `False` and run:

```shell
# Training on single CPU/GPU/Ascend devices
python tools/train.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

The training results (including checkpoint, performance of each epoch, and curve graph) will be saved in the directory configured by the `train.ckpt_save_dir` parameter in the YAML configuration file, which is set to `./tmp_rec` by default.

### Resuming Training From Checkpoint

If users expect to load the optimizer, learning rate, and other information of the model while starting or continue training, they can add `model.resume` to the corresponding local model path in the configuration file as follows, and start training:

```yaml
...
model:
  type: rec
  transform: null
  backbone:
    name: rec_resnet34
    pretrained: False
  neck:
    name: RNNEncoder
    hidden_size: 64
  head:
    name: CTCHead
    out_channels: *num_classes
  resume: /local_path_to_the_ckpt/model.ckpt
...
```

#### Mixed Precision Training

Some models (including CRNN, RARE, SVTR) support mixed precision training to accelerate training speed. Users can try setting the `system.amp_level` in the configuration file to `O2` to start mixed precision training, as shown in the following example:

```yaml
system:
  mode: 0
  distribute: True
  amp_level: O2  # Mixed precision training
  amp_level_infer: O2
  seed: 42
  log_interval: 100
  val_while_train: True
  drop_overflow_update: False
  ckpt_max_keep: 5
...
```

To disable mixed precision training, change `system.amp_level` to `O0`.

## Model Evaluation

To evaluate the accuracy of a trained model, users can use `tools/eval.py`. Please set the `ckpt_load_path` parameter in the `eval` section of the configuration file to the file path of the model checkpoint, and set `system.distribute` to False, as shown below:

```yaml
system:
  distribute: False # During evaluation stage, set to False
...
eval:
  ckpt_load_path: /local_path_to_the_ckpt/model.ckpt
```

and run

```shell
python tools/eval.py --config configs/rec/crnn/crnn_resnet34_ch.yaml
```

You will get a model evaluation result similar to the following:

```log
2023-06-16 03:41:20,237:INFO:Performance: {'acc': 0.821939, 'norm_edit_distance': 0.917264}
```

The number corresponding to `acc` is the accuracy of the model.

## Model Inference

Users can quickly obtain the inference results of the model by using the inference script. First, place the images in the same folder, and then execute:

```shell
python tools/infer/text/predict_rec.py --image_dir {dir_to_your_image_data} --rec_algorithm CRNN_CH --draw_img_save_dir inference_results
```

The results will be stored in `draw_img_save_dir/rec_results.txt`. Here are some examples:

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/e220ade5-89ae-47a4-927f-2c28941a5965" width=200 />
</p>
<p align="center">
  <em> cert_id.png </em>
</p>

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/d7cfee90-d586-4796-9ebf-b56872832e71" width=400 />
</p>
<p align="center">
  <em> doc_cn3.png </em>
</p>

You will get inference results similar to the following:

```text
cert_id.png 公民身份号码44052419
doc_cn3.png 马拉松选手不会为短暂的领先感到满意，而是永远在奔跑。
```
