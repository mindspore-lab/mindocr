### MindOCR推理

#### 支持模型列表

##### 文本检测

| 模型          | 链接                                                                                 | 来源        |
|-------------|------------------------------------------------------------------------------------|-----------|
| DBNet_Res18 | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar | PaddleOCR |
| DBNet_MV3   | https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar         | PaddleOCR |

##### 文本方向分类

| 模型  | 链接                                                                                 | 来源        |
|-----|------------------------------------------------------------------------------------|-----------|
| MV3 | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar | PaddleOCR |

##### 文本识别

| 模型   | 链接                                                                                 | 来源        |
|------|------------------------------------------------------------------------------------|-----------|
| CRNN | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar | PaddleOCR |
| SVTR | https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar         | PaddleOCR |

#### Benchmark

1. 测试数据

   数据集：ICDAR2019-LSVT

   下载链接：https://rrc.cvc.uab.es/?ch=16

   数据描述：30000张，街景图像，比如各种店铺招牌和地标等


2. 测试环境

   Device: Ascend310P

   MindX: 3.0.0

   CANN: 6.3

   CPU: Intel(R) Xeon(R) Gold 6148, 2.40GHz, 2x20 physical cores

3. 评估说明

    - 性能：包括从图像输入到结果输出的完整阶段，设置mindocr推理命令--save_log_dir参数，保存日志中会记录性能数据
    - 精度：参考评估脚本 mindocr/deploy/eval_utils/eval_script.py，输出结果包括文本检测的Precision、Recall和F-score，文本识别的Accuracy

4. 精度和性能评估结果

文本检测+方向分类+文本识别的端到端流水线评估结果如下：

| 检测        | 分类 | 识别 | Precision | Recall | F-score | Accuracy | FPS   |
| ----------- | ---- | ---- | --------- | ------ | ------- | -------- | ----- |
| DBNet_Res18 | /    | CRNN | 69.42%    | 55.01% | 61.38%  | 47.12%   | 38.59 |
| DBNet_Res18 | MV3  | CRNN | 69.42%    | 55.01% | 61.38%  | 46.85%   | 37.40 |
| DBNet_MV3   | /    | SVTR | 67.01%    | 56.34% | 61.21%  | 46.77%   | 46.65 |
| DBNet_MV3   | MV3  | SVTR | 67.01%    | 56.34% | 61.21%  | 46.45%   | 42.81 |