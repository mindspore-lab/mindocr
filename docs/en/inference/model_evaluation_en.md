English | [中文](../../cn/inference/model_evaluation_cn.md)
## Model Inference Evaluation
#### 1. Text detection
After inference, please use the following command to evaluate the results:
```shell
python deploy/eval_utils/eval_det.py \
		--gt_path=/path/to/det_gt.txt \
		--pred_path=/path/to/prediction/det_results.txt
```
#### 2. Text recognition
After inference, please use the following command to evaluate the results:
```shell
python deploy/eval_utils/eval_rec.py \
		--gt_path=/path/to/rec_gt.txt \
		--pred_path=/path/to/prediction/rec_results.txt \
		--character_dict_path=/path/to/xxx_dict.txt
```
Please note that **character_dict_path** is an optional parameter, and the default dictionary only supports English recognition.

When evaluating the PaddleOCR series models, please refer to [Third-Party Model Support List](./models_list_thirdparty_en.md) to use the corresponding dictionary.
