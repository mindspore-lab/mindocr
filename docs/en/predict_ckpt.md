# MindOCR detection and recognition prediction pipeline

This doc introduces how to run the detection and recognition prediction pipeline using MindOCR-trained ckpt files.

## 1. Pipeline model lists

| Text detection + text recognition pipeline | Datasets                                                          | Inference acc |
|--------------------------------------------|-------------------------------------------------------------------|---------------|
| DBNet+CRNN                                 | [ICDAR15](https://rrc.cvc.uab.es/?ch=4&com=downloads)<sup>*</sup> | 55.99%        |

> *We use Test Set in ICDAR15 Task 4.1.

## 2. Quick start

### 2.1 Dependency

| Environment | Version |
|-------------|---------|
| MindSpore   | >=1.9   |
| Python      | >=3.7   |


### 2.2 Argument configuration

Argument configuration includes two parts: 
- (1) yaml config file
- (2) args in `tools/predict/text/predict_system.py`

**Note that if you set the values of the args by (2), those args values will overwrite their counterparts in (1) yaml config file. 
Otherwise, the args values in (1) yaml config file will be used by default. You can also update the args values in yaml config file directly.**

#### (1) yaml config file

   Detection model and recognition model have one yaml config file respectively. Please pay attention to the `predict` module in both detection and recognition config files. The important args are listed below.

   ```yaml
   # yaml config file for detection model or recognition model
   ...
   predict:
     ckpt_load_path: tmp_det/best.ckpt              <--- args.det_ckpt_path (if set) overwrites it in det yaml, args.rec_ckpt_path (if set) overwrites it in rec yaml; or update it here directly
     dataset_sink_mode: False
     dataset:
       type: PredictDataset
       dataset_root: path/to/dataset_root           <--- args.raw_data_dir (if set) overwrites it in det yaml, args.crop_save_dir (if set) overwrites it in rec yaml; or update it here directly
       data_dir: ic15/det/test/ch4_test_images      <--- args.raw_data_dir (if set) overwrites it in det yaml, args.crop_save_dir (if set) overwrites it in rec yaml; or update it here directly
       sample_ratio: 1.0
       transform_pipeline:
         ...
       output_columns: [ 'img_path', 'image', 'raw_img_shape' ]
       num_columns_to_net: 1
     loader:
       shuffle: False
       batch_size: 1
       ...
   ```

#### (2) args list in prediction script `predict_system.py`
   | Argument          | Explanation                                                                                                        | Default                                    |
   |-------------------|-----------------------------------------| -------- |
   | raw_data_dir      | Directory of raw data to be predicted                                                                              | -                                       |
   | det_ckpt_path     | Path of detection model ckpt file                                                                                  | -                                       |
   | rec_ckpt_path     | Path of recognition model ckpt file                                                                                | -                                       |
   | det_config_path   | Path of detection model yaml config file                                                                           | 'configs/det/dbnet/db_r50_icdar15.yaml' |
   | rec_config_path   | Path of recognition model yaml config file                                                                         | 'configs/rec/crnn/crnn_resnet34.yaml'   |
   | crop_save_dir     | Directory for saving the cropped images after detection, i.e., **directory of input images for recognition model** | 'predict_result/crop'                   |
   | result_save_path  | Path for saving the pipeline prediction results                                                                    | 'predict_result/ckpt_pred_result.txt'   |


### 2.3 Prediction

   Run the following command to start the detection and recognition prediction pipeline. **Note that the args values below will overwrite their counterparts in yaml config file.**

   ```bash
   python tools/predict/text/predict_system.py \
                --raw_data_dir path/to/raw_data \
                --det_ckpt_path path/to/detection_ckpt \
                --rec_ckpt_path path/to/recognition_ckpt
   ```

### 2.4 Evaluation of prediction results
   After the prediction finishes, the results including image names, bounding boxes (`points`) and recognized texts (`transcription`) will be saved in `args.result_save_path`. The format of prediction results is shown below.
   ```text
   img_1.jpg	[{"transcription": "hello", "points": [600, 150, 715, 157, 714, 177, 599, 170]}, {"transcription": "world", "points": [622, 126, 695, 129, 694, 154, 621, 151]}, ...]
   img_2.jpg	[{"transcription": "apple", "points": [553, 338, 706, 318, 709, 342, 556, 362]}, ...]
   ...
   ```
   Prepare the **ground truth** file (in the same format as above) and **prediction results** file, and then run the following command to evaluate the prediction results.
   ```bash
   cd deploy/eval_utils
   python eval_pipeline.py --gt_path path/to/gt.txt --pred_path path/to/ckpt_pred_result.txt
   ```