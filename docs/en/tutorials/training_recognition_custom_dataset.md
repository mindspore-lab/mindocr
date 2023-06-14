English | [中文](../../cn/tutorials/training_recognition_custom_dataset_CN.md)

# Training Recognition Network with Custom Datasets

This document provides tutorials on how to train recognition networks using custom datasets, including the training of recognition networks in Chinese and English languages.

## Dataset preperation

Currently, MindOCR recognition network supports two input formats, namely
- `Common Dataset`：A file format that stores images and text files. It is read by [RecDataset](../../../mindocr/data/rec_dataset.py).
- `LMDB Dataset`: A file format provided by [LMDB](https://www.symas.com/lmdb). It is read by [LMDBDataset](../../../mindocr/data/rec_lmdb_dataset.py).

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

## Model Training And Evaluation

When all datasets and configuration files have been prepared, users can start training their custom models. Since the training methods for different models are different, users can refer to the `Model Training` and `Model Evaluation` sections of the documentation for the corresponding model.
