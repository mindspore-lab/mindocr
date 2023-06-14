[English](../../en/inference/convert_dynamic_en.md) | 中文

## 推理 - 模型Shape分档

#### 1 简介

根据提供的数据集，统计图像`height`和`width`的分布范围，离散的选择`batch size`、`height`、`width`组合，实现分档，再使用 [ATC](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000001.html) 或 [MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html) 进行模型转换。

#### 2 环境准备

| 环境     | 版本             |
|--------|----------------|
| Device | Ascend310/310P |
| Python | \>= 3.7        |

#### 3 模型准备

例如，需要先下载推理模型（
[检测](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) ，
[识别](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) ，
[分类](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar)
），使用 [paddle2onnx](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme.md) 工具转换得到以下ONNX模型。

| 模型类别 | 模型名称                                | 输入shape     |
|------|-------------------------------------|-------------|
| 检测   | ch_PP-OCRv3_det_infer.onnx          | -1,3,-1,-1  |
| 识别   | ch_PP-OCRv3_rec_infer.onnx          | -1,3,48,-1  |
| 分类   | ch_ppocr_mobile_v2.0_cls_infer.onnx | -1,3,48,192 |

#### 4 数据集准备

例如，数据集[ICDAR 2015：`Text Localization`](https://rrc.cvc.uab.es/?ch=4&com=downloads) ，下载需要先注册账号。

数据集准备请参考[tools/dataset_converters/converter.py](../../../tools/dataset_converters/convert.py) 数据格式化转换脚本，并按照[README_CN`文本检测/端到端文本检测`](../../../tools/dataset_converters/README_CN.md) 部分执行脚本。最终得到图像和对应的标注文件。

#### 5 分档工具使用

##### 5.1 自动分档调用示例

参考`deploy/models_utils/auto_scaling/converter.py`将模型转OM模型。
  ```
  # git clone https://github.com/mindspore-lab/mindocr
  # cd mindocr/deploy/models_utils/auto_scaling

  # 示例1：对batch size进行分档
  python converter.py --model_path=/path/to/ch_PP-OCRv3_rec_infer.onnx \
                      --dataset_path=/path/to/det_gt.txt
                      --input_shape=-1,3,48,192 \
                      --output_path=output

  输出结果为单个OM模型：ch_PP-OCRv3_rec_infer_dynamic_bs.om。
  ```
  ```
  # 示例2：对height和width进行分档
  python converter.py --model_path=/path/to/ch_PP-OCRv3_det_infer.onnx \
                      --dataset_path=/path/to/images \
                      --input_shape=1,3,-1,-1 \
                      --output_path=output

  输出结果为单个OM模型：ch_PP-OCRv3_det_infer_dynamic_hw.om。
  ```
  ```
  # 示例3：对batch szie、height和width进行分档
  python converter.py --model_path=/path/to/ch_PP-OCRv3_det_infer.onnx \
                      --dataset_path=/path/to/images \
                      --input_shape=-1,3,-1,-1 \
                      --output_path=output

  输出结果为多个OM模型：ch_PP-OCRv3_det_infer_dynamic_bs1_hw.om，ch_PP-OCRv3_det_infer_dynamic_bs4_hw.om，……，ch_PP-OCRv3_det_infer_dynamic_bs64_hw.om。
  ```
  ```
  # 示例4：不做分档
  python converter.py --model_path=/path/to/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                      --input_shape=4,3,48,192 \
                      --output_path=output

  输出结果为单个OM模型：ch_ppocr_mobile_v2.0_cls_infer_static.om。
  ```

需要适配脚本对应数据和模型参数：

| 参数名称        | 描述                                                |
|-------------|---------------------------------------------------|
| model_path  | 必选，需要转换的模型文件路径。                                   |
| data_path   | 非必选，检测模型为数据集图片路径，识别模型为标注文件路径，用户不传此参数会读取配置文件中分档数据。 |
| input_name  | 非必选，模型输入变量名，默认：x。                                 |
| input_shape | 必选，模型输入shape：NCHW，N、H、W支持分档。                      |
| backend     | 非必选，转换工具：atc或lite，默认：atc。                         |
| output_path | 非必选，输出模型保存路径，默认：./output。                         |
| soc_version | 非必选，Ascend310P3或Ascend310P，默认：Ascend310P3。        |


##### 5.2 `ATC`或`MindSpore Lite`单独使用示例

在`deploy/models_utils/auto_scaling/example`下给出了单独调用`ATC`或`MindSpore Lite`转换的示例。

  ```
  # ATC
  atc --model=/path/to/ch_ppocr_mobile_v2.0_cls_infer.onnx \
      --framework=5 \
      --input_shape="x:-1,3,48,192" \
      --input_format=ND \
      --dynamic_dims="1;4;8;16;32" \
      --soc_version=Ascend310P3 \
      --output=output \
      --log=error
  ```
  输出结果为单个OM模型：output.om。更多请参考：[ATC示例](../../../deploy/models_utils/auto_scaling/example/atc)

  ```
  # MindSpore Lite
  converter_lite  --modelFile=/path/to/ch_PP-OCRv3_det_infer.onnx \
      --fmk=ONNX \
      --configFile=lite_config.txt \
      --saveType=MINDIR \
      --NoFusion=false \
      --device=Ascend \
      --outputFile=output
  ```
  输出结果为单个OM模型：output.om。更多请参考：[MindSpore Lite示例](../../../deploy/models_utils/auto_scaling/example/mindspore_lite)

  注意：`MindSpore Lite`转换需要配置一个`lite_config.txt`文件, 如下所示：
  ```
  [ascend_context]
  input_format = NCHW
  input_shape = x:[1,3,-1,-1]
  dynamic_dims = [1248,640],[1248,672],...,[1280,768],[1280,800]
  ```

##### 5.3 自动分档配置文件

`limit_side_len`： 原始输入数据的width和height大小限制，超出范围按照比例进行压缩，可以调整数据的离散程度。

`strategy`：数据统计算法策略，支持mean_std和max_min两种算法，默认：mean_std。

    假设数据平均值：mean，标准差：sigma，最大值：max，最小值：min。

    mean_std：计算方式：[mean - n_std * sigma，mean + n_std * sigma]，n_std：3。

    max_min：计算方式：[min - (max - min)*expand_ratio/2，max + (max - min)*expand_ratio/2]， expand_ratio：0.2。

`width_range/height_range`： 对离散统计之后的width/height大小限制，超出将被过滤。

`interval`：分档间隔大小。

`max_scaling_num`：分档组合数上限。

`batch_choices`：默认的batch size范围。

`default_scaling`：用户不传入数据时，提供默认的分档数据。

##### 5.4 自动分档工具代码目录结构
```
auto_scaling
├── configs
│   └── auto_scaling.yaml
├── converter.py
├── example
│   ├── atc
│   │   ├── atc_dynamic_bs.sh
│   │   ├── atc_dynamic_hw.sh
│   │   └── atc_static.sh
│   └── mindspore_lite
│       ├── lite_dynamic_bs.sh
│       ├── lite_dynamic_bs.txt
│       ├── lite_dynamic_hw.sh
│       ├── lite_dynamic_hw.txt
│       ├── lite_static.sh
│       └── lite_static.txt
├── __init__.py
└── src
    ├── auto_scaling.py
    ├── backend
    │   ├── atc_converter.py
    │   ├── __init__.py
    │   └── lite_converter.py
    ├── __init__.py
    └── scale_analyzer
        ├── dataset_analyzer.py
        └── __init__.py
```
