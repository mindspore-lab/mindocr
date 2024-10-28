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
