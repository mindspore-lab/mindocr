# MindOCR在线推理

**关于在线推理：** 在线推理是基于原生MindSpore框架，通过加载模型文件，然后使用MindSpore API运行预测来进行推理。

与离线推理（在MindOCR中的“deploy/py_infer”中实现）相比，在线推理不需要对目标平台进行模型转换，可以直接在训练设备（例如Ascend 910）上运行。但是它需要安装重型AI框架，并且模型没有针对部署进行优化。

因此，在线推理更适合于演示和可视化评估模型对未知数据的泛化能力。

## 依赖关系和安装
与训练环境一致。

## 文本检测

要对输入图像或包含多个图像的目录运行文本检测，请执行
```shell
python tools/infer/text/predict_det.py  --image_dir {path_to_img or dir_to_imgs} --det_algorithm DB++
```

运行后，推理结果保存在`{args.draw_img_save_dir}/det_results.txt`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置，这里是一些示例结果。

示例1：
<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/ce136b92-f0aa-4a05-b689-9f60d0b40db1" width=480 />
</p>
<p align="center">
  <em> img_108.jpg的可视化结果</em>
</p>

其中保存的txt文件如下
```
img_108.jpg	[[[228, 440], [403, 413], [406, 433], [231, 459]], [[282, 280], [493, 252], [499, 293], [288, 321]], [[500, 253], [636, 232], [641, 269], [505, 289]], ...]
```

示例2：
<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/61066d4a-5922-471e-b702-2ea79c3cc525" width=480 />
</p>
<p align="center">
  <em>paper_sam.png的可视化结果</em>
</p>

其中保存的txt文件如下
```
paper_sam.png	[[[1161, 340], [1277, 340], [1277, 378], [1161, 378]], [[895, 335], [1152, 340], [1152, 382], [894, 378]], ...]
```

**注意事项：**
- 对于高分辨率的输入图像，请将`--det_limit_side_len`设置得更大，例如1280。`--det_limit_type`可以设置为“min”或“max”，其中“min”表示限制图像大小至少为`--det_limit_side_len`，“max”表示限制图像大小最多为`--det_limit_side_len`。

- 有关更多参数说明和用法，请运行`python tools/infer/text/predict_det.py -h`或查看`tools/infer/text/config.py`

- 目前，该脚本可以持续运行以避免动态形状问题并获得更好的性能。

### 支持的检测算法和网络

<center>

  |**算法名称**|**网络名称**|**语言**|
  | :------: | :------: | :------: |
  |DB | dbnet_resnet50 |英语|
  |DB++| dbnet_resnet50 |英语|
  |DB_MV3 | dbnet_mobilenetv3 |英语|
  |PSE | psenet_resnet152 |英语|

</center>

算法网络在`tools/infer/text/predict_det.py`中定义。

## 文本识别

要对输入图像或包含多个图像的目录运行文本识别，请执行
```shell
python tools/infer/text/predict_rec.py  --image_dir {path_to_img or dir_to_imgs} --rec_algorithm CRNN
```
运行后，推理结果保存在`{args.draw_img_save_dir}/rec_results.txt`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置。下面是一些结果的例子。

- 英文文本识别

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/fa8c5e4e-0e05-4c93-b9a3-6e0327c1609f" width=150 />
</p>
<p align="center">
  <em> word_1216.png </em>
</p>

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/8ec50bdf-ea6c-4bce-a799-2fdb8e9512b1" width=150 />
</p>
<p align="center">
  <em> word_1217.png </em>
</p>

识别结果：
```text
word_1216.png   coffee
word_1217.png   club
```

- 中文文本识别：

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

识别结果：
```text
cert_id.png 公民身份号码44052419
doc_cn3.png 马拉松选手不会为短暂的领先感到满意，而是永远在奔跑。
```

**注意事项：**
- 有关更多参数说明和用法，请运行`python tools/infer/text/predict_rec.py -h`或查看`tools/infer/text/config.py`
- 支持批量推理和单模推理。默认情况下启用批处理模式以提高速度。您可以通过`--rec_batch_size`设置批量大小。您还可以通过设置`--det_batch_mode` False在单一模式下运行，如果文本长度变化很大，这可能会提高准确性。

### 支持的识别算法和网络

<center>

  |**算法名称**|**网络名称**|**语言**|
  | :------: | :------: | :------: |
  | CRNN | crnn_resnet34 | 英语|
  | RARE | rare_resnet34 | 英语|
  | SVTR | svtr_tiny |英语|
  | CRNN_CH | crnn_resnet34_ch | 中文|
  | RARE_CH | rare_resnet34_ch | 中文|

</center>

算法网络在`tools/infer/text/predict_rec.py`中定义

目前，所列型号不支持空格字符识别。我们将很快予以支持。

## 文本检测与识别级联

要对输入图像或目录中的多个图像运行文本定位（即检测所有文本区域，然后识别每个文本区域），请运行：

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN
```
>注意：如果要可视化输入图像上的检测和识别结果，请设置`--visualize_output True`。

运行后，推理结果保存在`{args.draw_img_save_dir}/system_results.txt`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置。下面是一些结果的例子。

示例1：

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=600 />
</p>
<p align="center">
  <em> img_10.jpg检测和识别的可视化结果 </em>
</p>

其中保存的txt文件如下
```text
img_10.jpg	[{"transcription": "residential", "points": [[43, 88], [149, 78], [151, 101], [44, 111]]}, {"transcription": "areas", "points": [[152, 83], [201, 81], [202, 98], [153, 100]]}, {"transcription": "when", "points": [[36, 56], [101, 56], [101, 78], [36, 78]]}, {"transcription": "you", "points": [[99, 54], [143, 52], [144, 78], [100, 80]]}, {"transcription": "pass", "points": [[140, 54], [186, 50], [188, 74], [142, 78]]}, {"transcription": "by", "points": [[182, 52], [208, 52], [208, 75], [182, 75]]}, {"transcription": "volume", "points": [[199, 30], [254, 30], [254, 46], [199, 46]]}, {"transcription": "your", "points": [[164, 28], [203, 28], [203, 46], [164, 46]]}, {"transcription": "lower", "points": [[109, 25], [162, 25], [162, 46], [109, 46]]}, {"transcription": "please", "points": [[31, 18], [109, 20], [108, 48], [30, 46]]}]
```

示例2：

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c58fb182-32b0-4b73-b4fd-7ba393e3f397" width=480 />
</p>
<p align="center">
  <em> web_cvpr.png检测和识别的可视化结果 </em>
</p>

其中保存的txt文件如下
```text
web_cvpr.png	[{"transcription": "canada", "points": [[430, 148], [540, 148], [540, 171], [430, 171]]}, {"transcription": "vancouver", "points": [[263, 148], [420, 148], [420, 171], [263, 171]]}, {"transcription": "cvpr", "points": [[32, 69], [251, 63], [254, 174], [35, 180]]}, {"transcription": "2023", "points": [[194, 44], [256, 45], [255, 72], [194, 70]]}, {"transcription": "june", "points": [[36, 45], [110, 44], [110, 70], [37, 71]]}, {"transcription": "1822", "points": [[114, 43], [190, 45], [190, 70], [113, 69]]}]
```

**注意事项：**
1、如需更多参数说明和用法，请运行`python tools/infer/text/predict_system.py -h`或查看`tools/infer/text/config.py`

### 推理结果的评估

为了推理整个[ICDAR15](https://rrc.cvc.uab.es/?ch=4&com=downloads)测试集，请运行：
```
python tools/infer/text/predict_system.py --image_dir /path/to/icdar15/det/test_images  /
                                          --det_algorithm {DET_ALGO}    /
                                          --rec_algorithm {REC_ALGO}  /
                                          --det_limit_type min  /
                                          --det_limit_side_len 720
```
>注意：由于ICDAR15中的输入图像具有高分辨率（720x1280），因此我们将`det_limit_type`设置为'min'以获得更好的性能。

运行后，包括图像名称、边界框(`points`)和识别文本 (`transcription`)在内的结果将保存在`{args.draw_img_save_dir}/system_results.txt`中。预测结果格式如下：。

```text
img_1.jpg	[{"transcription": "hello", "points": [600, 150, 715, 157, 714, 177, 599, 170]}, {"transcription": "world", "points": [622, 126, 695, 129, 694, 154, 621, 151]}, ...]
img_2.jpg	[{"transcription": "apple", "points": [553, 338, 706, 318, 709, 342, 556, 362]}, ...]
   ...
```

准备 **ground truth** 文件（格式同上），可从`tools/dataset_converters`中的数据集转换脚本获取，运行以下命令对预测结果进行评估。
```bash
python deploy/eval_utils/eval_pipeline.py --gt_path path/to/gt.txt --pred_path path/to/system_results.txt
```

使用MindSpore 2.0rc1对Ascend 910上的文本定位推理结果的评估如下所示。
<center>

| Det. Algorithm| Rec. Algorithm |  Dataset     | Accuracy(%) | FPS (imgs/s) |
|---------|----------|--------------|---------------|-------|
| DBNet   | CRNN    | ICDAR15 | 57.82 | 4.86 |
| PSENet  | CRNN    | ICDAR15 | 47.91 | 1.65|
| PSENet (det_limit_side_len=1472 )  | CRNN    | ICDAR15 | 55.51 | 0.44 |
| DBNet++   | RARE | ICDAR15 | 59.17  | 3.47 |
| DBNet++   | SVTR | ICDAR15 | 64.42  | 2.49 |

</center>

**注意事项：**

1、目前在线推理流水线未进行效率优化，FPS仅用于模型间的比较。如果FPS是您的最高优先级，请参考[Ascend 310上的推断](https://github.com/mindspore-lab/mindocr/blob/main/docs/zh/inference/inference_tutorial.md)，这要快得多。

2、除非另有说明，所有实验均以`--det_limit_type`="min"和`--det_limit_side`=720运行。

3、SVTR在混合精度模式下运行（amp_level=O2），因为它针对O2进行了优化。

## 参数列表

所有CLI参数定义都可以通过`python tools/infer/text/predict_system.py -h`或`tools/infer/text/config.py`查看。

## 开发人员指南-如何添加新的推断模型

### 预处理

最佳预处理策略可能因模型而异，特别是对于调整大小设置（keep_ratio, padding等）。对于不同的任务，我们在`tools/infer/text/preprocess.py`中为每个模型定义了预处理管道。

如果发现默认的预处理管道或超参数不满足网络要求，请通过更改If-else条件或向`tools/infer/text/preprocess.py`中的`optimal_hparam` dict添加新的键值对进行扩展，其中key是算法名称，该值是用于目标网络推断的合适的超参数设置。

### 网络推理

在`predict_det.py`和`predict_rec.py`中的`algo_to_model_name` dict中定义了支持的算法及其相应的网络名称（可使用`list_model()` API进行检查）。

要添加新的检测模型进行推断，请在`algo_to_model_name` dict中添加一个新的键值对，其中键值是算法名称，该值是注册在`mindocr/models/{your_model}.py`中的相应网络名称。

默认情况下，模型权重将从`mindocr/models/{your_model}.py`中的pro-defined URL加载。如果要加载本地检查点，请将`--det_model_dir`或`--rec_model_dir`设置为本地检查点的路径或包含模型检查点的目录。

### 后处理程序
与预处理类似，每个算法的后处理方法可能会有所不同。每个算法的后处理方法在`tools/infer/text/postprocess.py`中定义。

如果您发现默认的后处理方法或超参数不满足模型需要，请扩展If-else条件或在`tools/infer/text/postprocess.py`中的`optimal_hparam` dict中添加一个新的键值对，其中键是算法名称，值是超参数设置。
