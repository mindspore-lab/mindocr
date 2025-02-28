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

### 文本方向分类

若图像中存在非正向的文字，可通过文本方向分类器对检测后的图像进行方向分类与矫正。若对输入图像运行文本方向分类与矫正，请执行
```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN  \
                                          --cls_algorithm M3
```
其中，参数`--cls_alorithm`默认配置为None，表示不执行文本方向分类，通过设置`--cls_alorithm`即可在文本检测识别流程中进行文本方向分类。执行过程中，文本方向分类器将对文本检测所得图像列表进行方向分类，并对非正向的图像进行方向矫正。以下为部分结果示例。

- 文本方向分类

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CRNN_t1.png" width=150 />
</p>
<p align="center">
  <em> word_01.png </em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CRNN_t2.png" width=150 />
</p>
<p align="center">
  <em> word_02.png </em>
</p>

分类结果：
```text
word_01.png   0     1.0
word_02.png   180   1.0
```
当前支持的文本方向分类网络为`mobilnet_v3`，可通过配置`--cls_algorithm`为`M3`进行设置，并通过`--cls_amp_level`与`--cls_model_dir`来设置文本方向分类器的自动混合精度与权重文件。当前已配置默认权重文件，该网络默认混合精度为`O0`，默认配置下方向分类支持`0`与`180`度，对于其他方向的分类我们将在未来予以支持。

<center>

  |**算法名称**|**网络名称**|**语言**|
  | :------: | :------: | :------: |
  | M3 | mobilenet_v3 | 中/英|

</center>

此外，可通过设置`--save_cls_result`为`True`可将文本方向分类结果保存至`{args.crop_res_save_dir}/cls_results.txt`中，其中`--crop_res_save_dir`是保存结果的目录。

有关更多参数说明和用法，请查看`tools/infer/text/config.py`

## 表格结构识别

要对输入图像或包含多个图像的目录运行表格结构识别，请执行
```shell
python tools/infer/text/predict_table_structure.py --image_dir {path_to_img or dir_to_imgs} --table_algorithm TABLE_MASTER
```

运行后，推理结果保存在`{args.draw_img_save_dir}`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置，这里是一些示例结果。

示例1：

样例图片为`configs/table/example.png`，运行结果如下：

<p align="center">
  <img src="https://github.com/user-attachments/assets/753588ff-3c24-4bf9-95c5-61c4c87d03f3" width=1000 />
</p>
<p align="center">
  <em> example_structure.png </em>
</p>

**注意事项：**
- 有关更多参数说明和用法，请运行`python tools/infer/text/predict_table_structure.py -h`或查看`tools/infer/text/config.py`

### 支持的表格结构识别算法和网络

<center>

  |   **算法名称**   |**网络名称**| **语言** |
  |:------------:| :------: |:------:|
  | table_master | table_resnet_extra |   不区分    |

</center>

算法网络在`tools/infer/text/predict_table_structure.py`中定义。

## 表格结构识别与文本检测识别级联

要对输入图像或目录中的多个图像运行表格识别（即识别表格结构后，结合文本检测识别的结果，识别出完整的表格内容），并恢复成csv文件，请运行：

```shell
python tools/infer/text/predict_table_recognition.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB_PPOCRv3  \
                                          --rec_algorithm SVTR_PPOCRv3_CH \
                                          --table_algorithm TABLE_MASTER
```

运行后，推理结果保存在`{args.draw_img_save_dir}`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置。下面是一些结果的例子。

示例1：

样例图片为`configs/table/example.png`，在线推理后，得到csv文件内容如下：
```txt
Parameter,Non-smokers Mean± SD or N (3),Smokers Mean ± SD or N (C)
N,24,
Age (y),69.1 ± 7.0,61.5 ± 9.3 +
Males/Females,24/0,11/0
Race White/Black,19/5,9/2
Weight (kg),97.8 ± 16.8,102.5 ± 23.4
BMII (kg/m*),32.6 ± 4.9,32.6 ± 6.6
Serum albumin (g/dL),3.8 ± 0.33,3.63 ± 0.30
Serum Creatinine (mg/dL),2.75 ± 1.21,1.80 ± 0.74 *
BUN (mg/dL),46.5 ± 25.6,38.3 ± 21.8
Hemoglobin (g/dL),13.3 ± 1.6,13.5 ± 2.4
24 hour urine protein (g/d),3393 ± 2522,4423 ± 4385
lathae)mm,28.9 ± 13.8,47.2 ± 34.8 *
Duration of diabetes (yr),15.7 ± 9.1,13.3 ± 9.0
Insulin use,15 (63%),6 (55%)
"Hemoglobin A, C (%)",7.57 ± 2.02,8.98 ± 2.93
Waist/Hip Ratio,1.00 ± 0.07,1.04 ± 0.07
Antihypertensive medications,4.3 ± 1.6,3.9 ± 1.9
A,21 (88%),8 (73%)
Total Cholesterol (mg/dL),184 ± 51,223 ± 87
LDL Cholesterol (mg/dL),100 ± 44,116 ± 24
HDL Cholesterol (mg/dL),42 ± 11.1,46 ± 11.4
,17 (71%),7 (64%)

```

**注意事项：**
1、如需更多参数说明和用法，请运行`python tools/infer/text/predict_table_recognition.py -h`或查看`tools/infer/text/config.py`

## 版面分析

要对输入图像或包含多个图像的目录运行版面分析，请执行
```shell
python tools/infer/text/predict_layout.py  --image_dir {path_to_img or dir_to_imgs} --layout_algorithm YOLOv8 --visualize_output True
```
运行后，推理结果保存在`{args.draw_img_save_dir}/det_results.txt`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置，这里是一些示例结果。

事例1:
<p align="center">
  <img src="https://github.com/user-attachments/assets/0cc501a8-5764-4b3a-8080-3dbc0f3ecb5e" width=480>
</p>
<p align="center">
  <em> PMC4958442_00003.jpg的可视化结果</em>
</p>

其中保存的layout_result.txt文件如下
```
{"image_id": 0, "category_id": 1, "bbox": [308.649, 559.189, 240.211, 81.412], "score": 0.98431}
{"image_id": 0, "category_id": 1, "bbox": [50.435, 673.018, 240.232, 70.262], "score": 0.98414}
{"image_id": 0, "category_id": 3, "bbox": [322.805, 348.831, 225.949, 203.302], "score": 0.98019}
{"image_id": 0, "category_id": 1, "bbox": [308.658, 638.657, 240.31, 70.583], "score": 0.97986}
{"image_id": 0, "category_id": 1, "bbox": [50.616, 604.736, 240.044, 70.086], "score": 0.9797}
{"image_id": 0, "category_id": 1, "bbox": [50.409, 423.237, 240.132, 183.652], "score": 0.97805}
{"image_id": 0, "category_id": 1, "bbox": [308.66, 293.918, 240.181, 47.497], "score": 0.97471}
{"image_id": 0, "category_id": 1, "bbox": [308.64, 707.13, 240.271, 36.028], "score": 0.97427}
{"image_id": 0, "category_id": 1, "bbox": [308.697, 230.568, 240.062, 43.545], "score": 0.96921}
{"image_id": 0, "category_id": 4, "bbox": [51.787, 100.444, 240.267, 273.653], "score": 0.96839}
{"image_id": 0, "category_id": 5, "bbox": [308.637, 74.439, 237.878, 149.174], "score": 0.96707}
{"image_id": 0, "category_id": 1, "bbox": [50.615, 70.667, 240.068, 22.0], "score": 0.94156}
{"image_id": 0, "category_id": 2, "bbox": [50.549, 403.5, 67.392, 12.85], "score": 0.92577}
{"image_id": 0, "category_id": 1, "bbox": [51.384, 374.84, 171.939, 10.736], "score": 0.76692}
```
其中，`image_id`为图像ID，`bbox`为检测出的边界框`[左上角的x坐标，右下角的y坐标，宽度，高度]`, `score`是检测的置信度，`category_id`的含义如下：
- `1: text`
- `2: title`
- `3: list`
- `4: table`
- `5: figure`

**注意事项：**
- 有关更多参数说明和用法，请运行`python tools/infer/text/predict_layout.py -h`或查看`tools/infer/text/config.py`

### 支持的检测算法和网络

<center>

  |**算法名称**|**网络名称**|**语言**|
  | :------: | :------: | :------: |
  |YOLOv8 | yolov8 |英语|

</center>

算法网络在`tools/infer/text/predict_layout.py`中定义。

## 端到端文档分析及恢复

要对输入图像或目录中的多个图像运行文档分析（即检测所有文本区域、表格区域、图像区域，并对这些区域进行文字识别，最终将结果按照图像原来的排版方式转换成Docx文件），请运行：

```shell
python tools/infer/text/predict_table_e2e.py --image_dir {path_to_img or dir_to_imgs} \
                                             --det_algorithm {DET_ALGO} \
                                             --rec_algorithm {REC_ALGO}
```
>注意：如果要可视化版面分析、表格识别和文字识别的结果，请设置`--visualize_output True`。

运行后，推理结果保存在`{args.draw_img_save_dir}/{img_name}_e2e_result.txt`中，其中`--draw_img_save_dir`是保存结果的目录，这是`./inference_results`的默认设置。下面是一些结果的例子。

示例1：

<p align="center">
  <img src="https://github.com/user-attachments/assets/f1578fda-c5c7-46ba-b446-d0dc65edf2d7"/>
</p>
<p align="center">
  <em> PMC4958442_00003.jpg转换成docx文件的效果 </em>
</p>

其中保存的txt文件如下
```text
{"type": "text", "bbox": [50.615, 70.667, 290.683, 92.667], "res": "tabley predictive value ofbasic clinical laboratory and suciode variables surney anc yea after tramphenins", "layout": "double"}
{"type": "table", "bbox": [51.787, 100.444, 292.054, 374.09700000000004], "res": "<html><body><table><thead><tr><td><b>sign factor</b></td><td><b>prediction valucofthe the</b></td><td><b>from difereness significance levelaf the</b></td></tr></thead><tbody><tr><td>gender</td><td>0027 0021</td><td>o442</td></tr><tr><td></td><td>00z44</td><td>0480</td></tr><tr><td>cause</td><td>tooza 0017</td><td>o547</td></tr><tr><td>cadaverieilizing donorst</td><td>0013 aont</td><td>0740</td></tr><tr><td>induction transplantation before dialysis</td><td>doattoos</td><td>0125</td></tr><tr><td>depleting antibodies monoclomalor cn immunosuppression with</td><td>doista09</td><td>0230</td></tr><tr><td>ititis</td><td>0029</td><td>aaso</td></tr><tr><td>status itional</td><td>0047 toots</td><td></td></tr><tr><td>townfrillage</td><td>non</td><td></td></tr><tr><td>transplantations number</td><td>toos 0017</td><td>o5s1</td></tr><tr><td>creatinine</td><td>02400g</td><td>caoor</td></tr><tr><td>pressure bload systolic</td><td>aidaloloss</td><td>aoz</td></tr><tr><td>pressure diastolic blood</td><td>dobetods</td><td>ass</td></tr><tr><td>hemoglobin</td><td>0044 0255t</td><td>caoor</td></tr><tr><td></td><td>004</td><td>caoor</td></tr></tbody></table></body></html>", "layout": "double"}
{"type": "text", "bbox": [51.384, 374.84, 223.32299999999998, 385.57599999999996], "res": "nanc rmeans more significant forecasting factor sign", "layout": "double"}
{"type": "title", "bbox": [50.549, 403.5, 117.941, 416.35], "res": "discussion", "layout": "double"}
{"type": "text", "bbox": [50.409, 423.237, 290.541, 606.889], "res": "determination of creatinine and hemoglobin level in the blood well aetho concentration of protein in the urine in one year atter kidney transplantation with the calculation of prognostic criterion predics the loss of renal allotransplant function in years fafter surgery advantages ff the method are the possibility oof quantitative forecasting of renal allotransplant losser which based not only its excretory function assessment but also on assessment other characteristics that may have important prognostic value and does not always directly correlate with changes in its excretors function in order the riskof death with transplant sfunctioning returntothe program hemodialysis the predictive model was implemented cabular processor excel forthe useofthe model litisquite enough the value ethel given indices calculation and prognosis will be automatically done in the electronic table figure 31", "layout": "double"}
{"type": "text", "bbox": [50.616, 604.736, 290.66, 674.822], "res": "the calculator designed by us has been patented chttpell napatentscomy 68339 sposib prognozuvannys vtrati funk caniskovogo transplanatchti and disnvailable on the in ternet chitpsolivad skillwond the accuract ot prediction of renal transplant function loss three years after transplantation was 92x", "layout": "double"}
{"type": "text", "bbox": [50.435, 673.018, 290.66700000000003, 743.28], "res": "progression of chronic renal dysfunctional the transplant accompanied the simultaneous losa the benefits of successful transplantation and the growth of problems due to immunosuppresson bosed on retrospective analysis nt resultsof treatment tofkidney transplantof the recipients with blood creatinine higher than d3 immold we adhere to the", "layout": "double"}
{"type": "figure", "bbox": [308.637, 74.439, 546.515, 223.613], "res": "./inference_results/example_figure_10.png", "layout": "double"}
{"type": "text", "bbox": [308.697, 230.568, 548.759, 274.113], "res": "figures the cnerhecadfmuthrnatical modeltor prognostication ofkidaey transplant function during the periodal three years after thetransplantation according oletectercipiolgaps after theoperation", "layout": "double"}
{"type": "text", "bbox": [308.66, 293.918, 548.841, 341.415], "res": "following principles in thecorrectionod immunisuppresion which allow decreasing the rateofs chronic dysfunctionof the transplant development or edecreasing the risk fof compliea tions incaeoflasof function", "layout": "double"}
{"type": "list", "bbox": [322.805, 348.831, 548.754, 552.133], "res": "wdo not prescribe hish doses steroids and do have the steroid pulse therapy cy do not increase the dose of received cyclosporine tacrolimus and stop medication ifthere isan increase in nephropathy tj continue immunosuppression with medicines ofmy cophenolic acid which are not nephrotoxic k4 enhance amonitoring of immunosuppression andpe vention infectious com cancel immunosuppression atreturning hemodi alysis treatment cancellation of steroids should done egradually sometimes for several months when thediscomfort eassociated transplant tempera ture main in the projection the transplanted kidney and hematurial short course of low doses of steroids administered orally of intravenously can be effective", "layout": "double"}
{"type": "text", "bbox": [308.649, 559.189, 548.86, 640.601], "res": "according to plasma concentration of creatinine the return hemodialvsis the patients were divided into groups ln the first group the creatinine concentration in blood plasma waso mmoly in the 2nd groun con centration in blood plasma was azlommaty and in the third group concentration in blood plasma was more than commolt", "layout": "double"}
{"type": "text", "bbox": [308.658, 638.657, 548.9680000000001, 709.24], "res": "dates or the return of transplant recipients with delaved rena transplant disfunction are largely dependent ion the psychological state ofthe patient severity of depression the desire to ensure the irreversibility the transplanted kidney dysfunction and fear that the dialysis will contribute to the deterioration of renal transplant function", "layout": "double"}
{"type": "text", "bbox": [308.64, 707.13, 548.911, 743.158], "res": "the survival rateof patients ofthe first group after return in hemodialysis was years and in the second and third groups respectively 53132 and28426 years", "layout": "double"}

```
其中，`type`为检测区域的类型，`bbox`为检测出的边界框`[左上角的x坐标，右下角的y坐标，宽度，高度]`, `res`是检测的结果内容。

**注意事项：**
1. 如需更多参数说明和用法，请运行`python tools/infer/text/predict_table_e2e.py -h`或查看`tools/infer/text/config.py`
2. 除了config.py中的参数，predict_table_e2e.py还接受如下参数：
<center>

  |   **参数名**   |**描述**| **默认值** |
  |:------------:| :------: |:------:|
  | layout | 版面分析任务 |   True    |
  | ocr | 文字识别任务 |   True    |
  | table | 表格识别任务 |   True    |
  | recovery | 转换成Docx任务 |   True    |

</center>

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
