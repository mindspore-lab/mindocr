# MindOCR Online Inference

**About Online Inference:** Online inference is to infer based on the native MindSpore framework by loading the model checkpoint file then running prediction with MindSpore APIs.

Compared to offline inference (which is implemented in `deploy/py_infer` in MindOCR), online inferece does not require model conversion for target platforms and can run directly on the training devices (e.g. Ascend 910). But it requires installing the heavy AI framework and the model is not optimized for deployment.

Thus, online inference is more suitable for demonstration and to visually evaluate model generalization ability on unseen data.

## Dependency and Installation

To be consistent with training environment.

## Text Detection

To run text detection on an input image or a directory containing multiple images, please execute

```shell
python tools/infer/text/predict_det.py  --image_dir {path_to_img or dir_to_imgs} --det_algorithm DB++
```

After running, the inference results will be saved in `{args.draw_img_save_dir}/det_results.txt`, where `--draw_img_save_dir` is the directory for saving  results and is set to `./inference_results` by default Here are some results for examples.

Example 1:
<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/ce136b92-f0aa-4a05-b689-9f60d0b40db1" width=480 />
</p>
<p align="center">
  <em> Visualization of text detection result on img_108.jpg</em>
</p>

, where the saved txt file is as follows
```
img_108.jpg	[[[228, 440], [403, 413], [406, 433], [231, 459]], [[282, 280], [493, 252], [499, 293], [288, 321]], [[500, 253], [636, 232], [641, 269], [505, 289]], ...]
```

Example 2:

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/61066d4a-5922-471e-b702-2ea79c3cc525" width=480 />
</p>
<p align="center">
  <em>Visualization of text detection result on paper_sam.png</em>
</p>

, where the saved txt file is as follows
```
paper_sam.png	[[[1161, 340], [1277, 340], [1277, 378], [1161, 378]], [[895, 335], [1152, 340], [1152, 382], [894, 378]], ...]
```

**Notes:**
- For input images with high resolution, please set `--det_limit_side_len` larger, e.g., 1280. `--det_limit_type` can be set as "min" or "max", where "min " means limiting the image size to be at least  `--det_limit_side_len`, "max" means limiting the image size to be at most `--det_limit_side_len`.

- For more argument illustrations and usage, please run `python tools/infer/text/predict_det.py -h` or view `tools/infer/text/config.py`

- Currently, this script runs serially to avoid dynamic shape issue and achieve better performance.


### Supported Detection Algorithms and Networks

<center>

  | **Algorithm Name** | **Network Name** | **Language** |
  | :------: | :------: | :------: |
  | DB  | dbnet_resnet50 | English |
  | DB++ | dbnetpp_resnet50 | English |
  | DB_MV3 | dbnet_mobilenetv3 | English |
  | PSE | psenet_resnet152 | English |

</center>

The algorithm-network mapping is defined in `tools/infer/text/predict_det.py`.

## Text Recognition

To run text recognition on an input image or a directory containing multiple images, please execute

```shell
python tools/infer/text/predict_rec.py  --image_dir {path_to_img or dir_to_imgs} --rec_algorithm CRNN
```
After running, the inference results will be saved in `{args.draw_img_save_dir}/rec_results.txt`, where `--draw_img_save_dir` is the directory for saving  results and is set to `./inference_results` by default. Here are some results for examples.

- English text recognition

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

Recognition results:
```text
word_1216.png   coffee
word_1217.png   club
```

- Chinese text recognition:

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

Recognition results:
```text
cert_id.png 公民身份号码44052419
doc_cn3.png 马拉松选手不会为短暂的领先感到满意，而是永远在奔跑。
```

**Notes:**
- For more argument illustrations and usage, please run `python tools/infer/text/predict_rec.py -h` or view `tools/infer/text/config.py`
- Both batch-wise and single-mode inference are supported. Batch mode is enabled by default for better speed. You can set the batch size via `--rec_batch_size`. You can also run in single-mode by set `--det_batch_mode` False, which may improve accuracy if the text length varies a lot.

### Supported Recognition Algorithms and Networks

<center>

  | **Algorithm Name** | **Network Name** | **Language** |
  | :------: | :------: | :------: |
  | CRNN | crnn_resnet34 | English |
  | RARE | rare_resnet34 | English |
  | SVTR | svtr_tiny | English|
  | CRNN_CH | crnn_resnet34_ch | Chinese |
  | RARE_CH | rare_resnet34_ch | Chinese |

</center>

The algorithm-network mapping is defined in `tools/infer/text/predict_rec.py`

Currently, space char recognition is not supported for the listed models. We will support it soon.

## Text Detection and Recognition Concatenation

To run text spoting (i.e., detect all text regions then recognize each of them) on an input image or multiple images in a directory, please run:

```shell
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB++  \
                                          --rec_algorithm CRNN
```
> Note: set `--visualize_output True` if you want to visualize the detection and recognition results on the input image.

After running, the inference results will be saved in `{args.draw_img_save_dir}/system_results.txt`,  where `--draw_img_save_dir` is the directory for saving  results and is set to `./inference_results` by default. Here are some results for examples.

Example 1:

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c1f53970-8618-4039-994f-9f6dc1eee1dd" width=600 />
</p>
<p align="center">
  <em> Visualization of text detection and recognition result on img_10.jpg </em>
</p>

, where the saved txt file is as follows
```text
img_10.jpg	[{"transcription": "residential", "points": [[43, 88], [149, 78], [151, 101], [44, 111]]}, {"transcription": "areas", "points": [[152, 83], [201, 81], [202, 98], [153, 100]]}, {"transcription": "when", "points": [[36, 56], [101, 56], [101, 78], [36, 78]]}, {"transcription": "you", "points": [[99, 54], [143, 52], [144, 78], [100, 80]]}, {"transcription": "pass", "points": [[140, 54], [186, 50], [188, 74], [142, 78]]}, {"transcription": "by", "points": [[182, 52], [208, 52], [208, 75], [182, 75]]}, {"transcription": "volume", "points": [[199, 30], [254, 30], [254, 46], [199, 46]]}, {"transcription": "your", "points": [[164, 28], [203, 28], [203, 46], [164, 46]]}, {"transcription": "lower", "points": [[109, 25], [162, 25], [162, 46], [109, 46]]}, {"transcription": "please", "points": [[31, 18], [109, 20], [108, 48], [30, 46]]}]
```

Example 2:

<p align="center">
  <img src="https://github.com/SamitHuang/mindocr-1/assets/8156835/c58fb182-32b0-4b73-b4fd-7ba393e3f397" width=480 />
</p>
<p align="center">
  <em> Visualization of text detection and recognition result on web_cvpr.png </em>
</p>

, where the saved txt file is as follows

```text
web_cvpr.png	[{"transcription": "canada", "points": [[430, 148], [540, 148], [540, 171], [430, 171]]}, {"transcription": "vancouver", "points": [[263, 148], [420, 148], [420, 171], [263, 171]]}, {"transcription": "cvpr", "points": [[32, 69], [251, 63], [254, 174], [35, 180]]}, {"transcription": "2023", "points": [[194, 44], [256, 45], [255, 72], [194, 70]]}, {"transcription": "june", "points": [[36, 45], [110, 44], [110, 70], [37, 71]]}, {"transcription": "1822", "points": [[114, 43], [190, 45], [190, 70], [113, 69]]}]
```


**Notes:**
1. For more argument illustrations and usage, please run `python tools/infer/text/predict_system.py -h` or view `tools/infer/text/config.py`

## Layout Analysis

To run layout analysis on an input image or a directory containing multiple images, please execute
```shell
python tools/infer/text/predict_layout.py  --image_dir {path_to_img or dir_to_imgs} --layout_algorithm YOLOv8 --visualize_output True
```
After running, the inference results will be saved in `{args.draw_img_save_dir}/det_results.txt`, where `--draw_img_save_dir` is the directory for saving  results and is set to `./inference_results` by default Here are some results for examples.

Example 1:
<p align="center">
  <img src="../../../configs/layout/yolov8/images/result.png" width=480>
</p>
<p align="center">
  <em> Visualization of layout analysis result on PMC4958442_00003.jpg</em>
</p>

, where the saved layout_result.txt file is as follows
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
In this file, `image_id` is the image ID, `bbox` is the detected bounding box `[x-coordinate of the top-left corner, y-coordinate of the bottom-right corner, width, height]`, `score` is the detection confidence, and `category_id` has the following meanings:
- `1: text`
- `2: title`
- `3: list`
- `4: table`
- `5: figure`

**Notes:**
- For more argument illustrations and usage, please run `python tools/infer/text/predict_layout.py -h` or view `tools/infer/text/config.py`

### Supported Detection Algorithms and Networks

<center>

  | **Algorithm Name** | **Network Name** | **Language** |
  | :------: | :------: | :------: |
  |YOLOv8 | yolov8 |English|

</center>

The algorithm-network mapping is defined in `tools/infer/text/predict_layout.py`.

### Evaluation of the Inference Results

To infer on the whole [ICDAR15](https://rrc.cvc.uab.es/?ch=4&com=downloads) test set, please run:
```
python tools/infer/text/predict_system.py --image_dir /path/to/icdar15/det/test_images  /
                                          --det_algorithm {DET_ALGO}    /
                                          --rec_algorithm {REC_ALGO}  /
                                          --det_limit_type min  /
                                          --det_limit_side_len 720
```
> Note: Here we set`det_limit_type` as `min` for better performance, due to the input image in ICDAR15 is of high resolution (720x1280).

After running, the results including image names, bounding boxes (`points`) and recognized texts (`transcription`) will be saved in `{args.draw_img_save_dir}/system_results.txt`. The format of prediction results is shown as follows.

```text
img_1.jpg	[{"transcription": "hello", "points": [600, 150, 715, 157, 714, 177, 599, 170]}, {"transcription": "world", "points": [622, 126, 695, 129, 694, 154, 621, 151]}, ...]
img_2.jpg	[{"transcription": "apple", "points": [553, 338, 706, 318, 709, 342, 556, 362]}, ...]
   ...
```

Prepare the **ground truth** file (in the same format as above), which can be obtained from the dataset conversion script in `tools/dataset_converters`, and run the following command to evaluate the prediction results.

```bash
python deploy/eval_utils/eval_pipeline.py --gt_path path/to/gt.txt --pred_path path/to/system_results.txt
```

Evaluation of the text spotting inference results on Ascend 910 with MindSpore 2.0rc1 are shown as follows.

<center>

| Det. Algorithm| Rec. Algorithm |  Dataset     | Accuracy(%) | FPS (imgs/s) |
|---------|----------|--------------|---------------|-------|
| DBNet   | CRNN    | ICDAR15 | 57.82 | 4.86 |
| PSENet  | CRNN    | ICDAR15 | 47.91 | 1.65|
| PSENet (det_limit_side_len=1472 )  | CRNN    | ICDAR15 | 55.51 | 0.44 |
| DBNet++   | RARE | ICDAR15 | 59.17  | 3.47 |
| DBNet++   | SVTR | ICDAR15 | 64.42  | 2.49 |

</center>

**Notes:**
1. Currently, online inference pipeline is not optimized for efficiency, thus FPS is only for comparison between models. If FPS is your highest priority, please refer to [Inference on Ascend 310](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/inference/inference_tutorial.md), which is much faster.
2. Unless extra inidication, all experiments are run with `--det_limit_type`="min" and `--det_limit_side`=720.
3. SVTR is run in mixed precision mode (amp_level=O2) since it is optimized for O2.

## Table Structure Recognition

To run table structure recognition on an input image or multiple images in a directory, please run:

```shell
python tools/infer/text/predict_table_structure.py --image_dir {path_to_img or dir_to_imgs} --table_algorithm TABLE_MASTER
```

After running, the inference results will be saved in `{args.draw_img_save_dir}`, where `--draw_img_save_dir` is the directory for saving  results and is set to `./inference_results` by default. Here are some results for examples.

Example 1：

The sample image is `configs/table/example.png`. The inference result is as follows:

<p align="center">
  <img src="../../../configs/table/example_structure.png" width=1000 />
</p>
<p align="center">
  <em> example_structure.png </em>
</p>

**Notes:**
1. For more argument illustrations and usage, please run `python tools/infer/text/predict_table_structure.py -h` or view `tools/infer/text/config.py`

### Supported Table Structure Recognition Algorithms and Networks

<center>

  | **Model Name** |    **Backbone**    | **Language** |
  |:--------------:|:------------------:|:------------:|
  |  table_master  | table_resnet_extra |  universal   |

</center>

The algorithm-network mapping is defined in `tools/infer/text/predict_table_structure.py`.

## Table Structure Recognition and Text Detection Recognition Concatenation

To run table recognition on an input image or multiple images in a directory (i.e., recognize the table structure first, then combine the results of text detection and recognition to recognize the complete table content), and recovery to CSV files, please run:
```shell
python tools/infer/text/predict_table_recognition.py --image_dir {path_to_img or dir_to_imgs} \
                                          --det_algorithm DB_PPOCRv3  \
                                          --rec_algorithm SVTR_PPOCRv3_CH \
                                          --table_algorithm TABLE_MASTER
```

After running, the inference results will be saved in `{args.draw_img_save_dir}`, where `--draw_img_save_dir` is the directory for saving  results and is set to `./inference_results` by default. Here are some results for examples.

Example 1：

The sample image is `configs/table/example.png`. After online inference, the content of the CSV file is as follows:

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

**Notes:**
1. For more argument illustrations and usage, please run `python tools/infer/text/predict_table_recognition.py -h` or view `tools/infer/text/config.py`

## Argument List

All CLI argument definition can be viewed via `python tools/infer/text/predict_system.py -h` or reading `tools/infer/text/config.py`.


## Developer Guide - How to Add a New Model for Inference

### Preprocessing

The optimal preprocessing strategy can vary from model to model, especially for the resize setting (keep_ratio, padding, etc). We define the preprocessing pipeline for each model in `tools/infer/text/preprocess.py` for different tasks.

If you find the default preprocessing pipeline or hyper-params does not meet the network requirement, please extend by changing the if-else conditions or adding a new key-value pair to the `optimal_hparam` dict in `tools/infer/text/preprocess.py`, where key is the algorithm name and the value is the suitable hyper-param setting for the target network inference.

### Network Inference

Supported alogirhtms and their corresponding network names (which can be checked by using the `list_model()` API) are defined in the `algo_to_model_name` dict in `predict_det.py` and `predict_rec.py`.

To add a new detection model for inference, please add a new key-value pair to `algo_to_model_name` dict, where the key is an algorithm name and the value is the corresponding network name registered in `mindocr/models/{your_model}.py`.

By default, model weights will be loaded from the pro-defined URL in `mindocr/models/{your_model}.py`. If you want to load a local checkpoint instead, please set `--det_model_dir` or `--rec_model_dir` to the path of your local checkpoint or the directory containing a model checkpoint.

### Postproprocess

Similar to preprocessing, the postprocessing method for each algorithm can vary. The postprocessing method for each algorithm is defined in `tools/infer/text/postprocess.py`.

If you find the default postprocessing method or hyper-params does not meet the model need, please extend the if-else conditions or add a new key-value pair  to the `optimal_hparam` dict in `tools/infer/text/postprocess.py`, where the key is an algorithm name and the value is the hyper-param setting.
