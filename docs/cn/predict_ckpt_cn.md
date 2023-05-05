# MindOCR串联推理

本文档介绍如何使用MindOCR训练出来的ckpt文件进行文本检测+文本识别的串联推理。

## 1. 支持的串联模型组合

| 文本检测+文本识别模型组合 | 数据集                                                               | 推理精度    |
|---------------|-------------------------------------------------------------------|---------|
| DBNet+CRNN    | [ICDAR15](https://rrc.cvc.uab.es/?ch=4&com=downloads)<sup>*</sup> | 55.99%  |

> *此处用于推理的是ICDAR15 Task 4.1中的Test Set

## 2. 快速开始

### 2.1 环境配置

| 环境        | 版本    |
|-----------|-------|
| MindSpore | >=1.9 |
| Python    | >=3.7 |


### 2.2 参数配置

参数配置包含两部分：
- （1）模型yaml配置文件
- （2）推理脚本`tools/predict/text/predict_system.py`中的args参数。

**注意：如果在（2）中传入args参数值，则会覆盖（1）yaml配置文件中的相应参数值；否则，将会使用yaml配置文件中的默认参数值，您可以手动更新yaml配置文件中的参数值。**

#### (1) yaml配置文件

   检测模型和识别模型各有一个yaml配置文件。请重点关注这**两个**文件中`predict`模块内的内容，重点参数如下。

   ```yaml
   # 检测模型或识别模型的yaml配置文件
   ...
   predict:
     ckpt_load_path: tmp_det/best.ckpt              <--- args.det_ckpt_path覆盖检测yaml, args.rec_ckpt_path覆盖识别yaml; 或手动更新该值
     dataset_sink_mode: False
     dataset:
       type: PredictDataset
       dataset_root: path/to/dataset_root           <--- args.raw_data_dir覆盖检测yaml, args.crop_save_dir覆盖识别yaml; 或手动更新该值
       data_dir: ic15/det/test/ch4_test_images      <--- args.raw_data_dir覆盖检测yaml, args.crop_save_dir覆盖识别yaml; 或手动更新该值
       sample_ratio: 1.0
       transform_pipeline:
         ...
       output_columns: [ 'img_path', 'image', 'raw_img_shape' ]
     loader:
       shuffle: False
       batch_size: 1
       ...
   ```

#### (2) 推理脚本`predict_system.py`的args参数列表

   | 参数名            | 含义                                   | 默认值                                     |
   |--------------------------------------|-----------------------------------------| -------- |
   | raw_data_dir   | 待预测数据的文件夹                            | -                                       |
   | det_ckpt_path  | 检测模型ckpt文件路径                         | -                                       |
   | rec_ckpt_path  | 识别模型ckpt文件路径                         | -                                       |
   | det_config_path | 检测模型yaml配置文件路径                       | 'configs/det/dbnet/db_r50_icdar15.yaml' |
   | rec_config_path | 识别模型yaml配置文件路径                       | 'configs/rec/crnn/crnn_resnet34.yaml'   |
   | crop_save_dir  | 串联推理中检测后裁剪图片的保存文件夹，**即识别模型读取图片的文件夹** | 'predict_result/crop'                   |
   | result_save_path | 串联推理结果保存路径                           | 'predict_result/ckpt_pred_result.txt'   |


### 2.3 推理

   运行以下命令，开始串联推理。**以下传入的参数值将覆盖yaml文件中的对应参数值。**

   ```bash
   python tools/predict/text/predict_system.py \
                --raw_data_dir path/to/raw_data \
                --det_ckpt_path path/to/detection_ckpt \
                --rec_ckpt_path path/to/recognition_ckpt
   ```

### 2.4 精度评估

   推理完成后，图片名、文字检测框(`points`)和识别的文字(`trancription`)将保存在args.result_save_path。推理结果文件格式示例如下：
   ```text
   img_1.jpg	[{"transcription": "hello", "points": [600, 150, 715, 157, 714, 177, 599, 170]}, {"transcription": "world", "points": [622, 126, 695, 129, 694, 154, 621, 151]}, ...]
   img_2.jpg	[{"transcription": "apple", "points": [553, 338, 706, 318, 709, 342, 556, 362]}, ...]
   ...
   ```
   
   准备好串联推理图片的**ground truth文件**（格式与上述推理结果文件一致）和**推理结果文件**后，执行以下命令，开始对串联推理进行精度评估。
   ```bash
   cd deploy/eval_utils
   python eval_pipeline.py --gt_path path/to/gt.txt --pred_path path/to/ckpt_pred_result.txt
   ```