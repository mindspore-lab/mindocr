English | [中文](../../cn/inference/convert_dynamic_cn.md)

## Inference - Model Shape Scaling

#### 1 Introduction

According to the provided dataset, the distribution range of `height` and `width` is statistically counted, and the `batch size`, `height`, and `width` combination is selected discretely to achieve auto-scaling, and then [ATC](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000001.html) or [MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html) is used for model conversion.

#### 2 Environment

| Environment | Version        |
|-------------|----------------|
| Device      | Ascend310/310P |
| Python      | \>= 3.7        |

#### 3 Example Model

E.g. You need to download the inference model first(
[detection](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) ,
[recognition](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) ,
[classification](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar)
), then use [paddle2onnx](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme.md) to get following ONNX models.

| Model Type     | Model Name                          | Input Shape |
|----------------|-------------------------------------|-------------|
| detection      | ch_PP-OCRv3_det_infer.onnx          | -1,3,-1,-1  |
| recognition    | ch_PP-OCRv3_rec_infer.onnx          | -1,3,48,-1  |
| classification | ch_ppocr_mobile_v2.0_cls_infer.onnx | -1,3,48,192 |

#### 4 Example Dataset

E.g. Dataset of [ICDAR 2015: `Text Localization`](https://rrc.cvc.uab.es/?ch=4&com=downloads) , you need to register an account first.

Dataset preparation refer to format conversion script [tools/dataset_converters/converter.py](../../../tools/dataset_converters/convert.py), and execute the script according to the [README`Text Detection/Spotting Annotation`](../../../tools/dataset_converters/README.md) section.
Finally, you get the images and corresponding annotation file.

#### 5 Auto-Scaling Tool

(1) Example of auto-scaling

Refer to `deploy/models_utils/auto_scaling/converter.py` to convert the model to OM model.
  ```
  # git clone https://github.com/mindspore-lab/mindocr
  # cd mindocr/deploy/models_utils/auto_scaling

  # e.g 1: auto-scaling of batch size
  python converter.py --model_path=/path/to/ch_PP-OCRv3_rec_infer.onnx \
                      --dataset_path=/path/to/det_gt.txt
                      --input_shape=-1,3,48,192 \
                      --output_path=output

  The output result is an OM model: ch_PP-OCRv3_rec_infer_dynamic_bs.om.
  ```
  ```
  # e.g 2: auto-scaling of height and width
  python converter.py --model_path=/path/to/ch_PP-OCRv3_det_infer.onnx \
                      --dataset_path=/path/to/images \
                      --input_shape=1,3,-1,-1 \
                      --output_path=output

  The output result is an OM model: ch_PP-OCRv3_det_infer_dynamic_hw.om.
  ```
  ```
  # e.g 3: auto-scaling of batch size、height and width
  python converter.py --model_path=/path/to/ch_PP-OCRv3_det_infer.onnx \
                      --dataset_path=/path/to/images \
                      --input_shape=-1,3,-1,-1 \
                      --output_path=output

  The output results are multiple OM models: ch_PP-OCRv3_det_infer_dynamic_bs1_hw.om, ch_PP-OCRv3_det_infer_dynamic_bs4_hw.om, ..., ch_PP-OCRv3_det_infer_dynamic_bs64_hw.om.
  ```
  ```
  # e.g 4: no auto-scaling
  python converter.py --model_path=/path/to/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                      --input_shape=4,3,48,192 \
                      --output_path=output

  The output result is an OM model: ch_ppocr_mobile_v2.0_cls_infer_static.om.
  ```

You need to adapt the corresponding data and model parameters to the script:

| Parameter   | description                                                                                                                                                                                |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_path  | Required, the path of the model file that needs to be converted.                                                                                                                           |
| data_path   | Not required, the detection model is the image path of the images, the recognition model is the path of the annotation file, and the default data will be used if not pass this parameter. |
| input_name  | Not required, model input variable name, default: x.                                                                                                                                       |
| input_shape | Required, model input shape: NCHW, N、H、W support auto-scaling.                                                                                                                             |
| backend     | Not required, converter backend: atc or lite, default: atc.                                                                                                                                |
| output_path | Not required, output model save path, default: ./output.                                                                                                                                   |
| soc_version | Not required, Ascend310P3 or Ascend310P, default: Ascend310P3.                                                                                                                             |


(2) `ATC` or `MindSpore Lite` use examples

Several examples of individual calls to `ATC` or `MindSpore Lite` conversion are given under `deploy/models_utils/auto_scaling/example`.

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
  The output result is an OM model: output.om. More examples refer to: [ATC examples](../../../deploy/models_utils/auto_scaling/example/atc)

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
  The output result is an OM model: output.om. More examples refer to: [MindSpore Lite examples](../../../deploy/models_utils/auto_scaling/example/mindspore_lite)

  Note: `MindSpore Lite` conversion need a `lite_config.txt` file, as follows:
  ```
  [ascend_context]
  input_format = NCHW
  input_shape = x:[1,3,-1,-1]
  dynamic_dims = [1248,640],[1248,672],...,[1280,768],[1280,800]
  ```

(3) Introduction to config file

`limit_side_len`: The width and height size limits of the original input data, which are compressed proportionally if out of range, can adjust the degree of discreteness of the data.

`strategy`: Data statistics algorithm strategy, support mean_std and max_min, default: mean_std.

    suppose that data mean: mean, standard deviation: sigma, data max: max, data min: min.

    mean_std: calculation formula: [mean - n_std * sigma, mean + n_std * sigma], n_std: 3.

    max_min: calculation formula: [min - (max - min)*expand_ratio/2, max + (max - min)*expand_ratio/2], expand_ratio: 0.2.

`width_range/height_range`: The width/height size limit after discrete statistics will be filtered if exceeded.

`interval`: Auto-scaling interval size.

`max_scaling_num`: Auto-scaling combination num limit.

`batch_choices`: The default batch size range.

`default_scaling`: Dataset parameter not exists, will provide default auto-scaling data.

(4) Auto-scaling code structure
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
