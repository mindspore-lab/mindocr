## Inference - Dynamic Shape Scaling

### 1. Introduction

In some inference scenarios, such as object recognition after detection, the input batch size and image size of the
recognition network are not fixed because the number of object and the size of the object are not fixed. If each
inference is calculated according to the maximum batch size or maximum image size, it will cause a waste of computing
resources.

Therefore, you can set some candidate values during model conversion, and resize to the best matching candidate value
during inference, thereby improving performance. Users can manually select these candidate values empirically, or they
can be statistically derived from the dataset.

This tool integrates the function of dataset statistics, can count the appropriate combination of `batch size`, `height`
and `width` as candidate values, and encapsulates the model conversion tool, thus realizing the automatic model shape
scaling.

### 2. Environment

Please refer to [Environment Installation](environment.md) to install MindSpore Lite environment.

### 3. Model

Currently, ONNX model files are supported, and by MindSpore Lite, they are automatically shape scaling and
converted to MIndIR model files.

Please make sure that the input model is the dynamic shape version. For example, if the text detection model needs to
shape scaling for H and W, make sure that at least the H and W axes are dynamic, and the shape can be `(1,3,-1,-1)` and
`(-1,3,- 1,-1) `etc.

### 4. Dataset

Two types of data are supported:

1. Image folder

   - This tool will read all the images in the folder, record `height` and `width`, and count suitable candidate values

   - Suitable for text detection and text recognition models

2. Annotation file for text detection

   - Refer to [converter](../datasets/converters.md), which is the annotation file output when the
     parameter `task` is `det`

   - This tool will read the coordinates of the text box marked under each image, record `height` and `width`, and the
     number of boxes as `batch size`, and count suitable candidate values

   - Suitable for text recognition models

#### 5. Usages

`cd deploy/models_utils/auto_scaling`

##### 5.1 Command example

- auto shape scaling for batch size

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --dataset_path=/path/to/det_gt.txt
    --input_shape=-1,3,48,192 \
    --output_path=output
```

The output is a single MindIR model: `model_dynamic_bs.mindir`

- auto shape scaling for height and width

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --dataset_path=/path/to/images \
    --input_shape=1,3,-1,-1 \
    --output_path=output
```

The output is a single MindIR model: `model_dynamic_hw.mindir`

- auto shape scaling for batch size, height and width

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --dataset_path=/path/to/images \
    --input_shape=-1,3,-1,-1 \
    --output_path=output
```

The output result is multiple OM models, combining multiple different batch sizes, and each model uses the same dynamic
image size：`model_dynamic_bs1_hw.mindir`, `model_dynamic_bs4_hw.mindir`, ......

- no shape scaling

```shell
python converter.py \
    --model_path=/path/to/model.onnx \
    --input_shape=4,3,48,192 \
    --output_path=output
```

The output is a single MindIR model: `model_static.mindir`

##### 5.2 Parameter Details

| Name        | Default     | Required | Description                                      |
|:------------|:------------|:---------|:-------------------------------------------------|
| model_path  | None        | Y        | Path to model file                               |
| input_shape | None        | Y        | model input shape, NCHW format                   |
| data_path   | None        | N        | Path to image folder or annotation file          |
| input_name  | x           | N        | model input name                                 |
| backend     | lite         | N        | converter backend, lite or acl                  |
| output_path | ./output    | N        | Path to output model                             |
| soc_version | Ascend310P3 | N        | soc_version for Ascend，Ascend310P3 or Ascend310 |

##### 5.3 Configuration file

In addition to the above command line parameters, there are some parameters in
[auto_scaling.yaml](https://github.com/mindspore-lab/mindocr/tree/main/deploy/models_utils/auto_scaling/configs/auto_scaling.yaml) to describe the statistics of
the dataset, You can modify it yourself if necessary:

- limit_side_len

  The size limit of `height` and `width` of the original input data, if it exceeds the range, it will be compressed
  according to the ratio, and the degree of discreteness of the data can be adjusted.

- strategy

  Data statistics algorithm strategy, supports `mean_std` and `max_min` two algorithms, default: `mean_std`.

  - mean_std

  ```
  mean_std = [mean - n_std * sigma，mean + n_std * sigma]
  ```

  - max_min

  ```
  max_min = [min - (max - min) * expand_ratio / 2，max + (max - min) * expand_ratio / 2]
  ```

- width_range/height_range

  For the width/height size limit after discrete statistics, exceeding will be filtered.

- interval

  Interval size, such as some networks may require that the input size must be a multiple of 32.

- max_scaling_num

  The maximum number of discrete values for shape scaling .

- batch_choices

  The default batch size value, if the data_path uses an image folder, the batch size information cannot be counted, and
  the default value will be used.

- default_scaling

  If user does not use data_path, provide default `height` and `width` discrete values for shape scaling .
