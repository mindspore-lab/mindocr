[English](../../en/inference/model_evaluation_en.md) | 中文

## 模型推理精度评估

### 1. 文本检测

完成推理后，使用以下命令评估检测结果：
```shell
python deploy/eval_utils/eval_det.py \
		--gt_path=/path/to/det_gt.txt \
		--pred_path=/path/to/prediction/det_results.txt
```

### 2. 文本识别

完成推理后，使用以下命令评估识别结果：

```shell
python deploy/eval_utils/eval_rec.py \
		--gt_path=/path/to/rec_gt.txt \
		--pred_path=/path/to/prediction/rec_results.txt \
		--character_dict_path=/path/to/xxx_dict.txt
```

请注意，character_dict_path是可选参数，默认字典仅支持数字和英文小写。

在评估PaddleOCR或MMOCR系列模型时，请参照[第三方模型支持列表](./models_list_thirdparty_cn.md)使用对应字典。
