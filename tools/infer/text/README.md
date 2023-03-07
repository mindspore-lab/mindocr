### MindOCR推理

#### 简介

MindOCR的推理工具集成了文本检测、角度分类和文字识别模块，实现了端到端的OCR推理过程，并采用流水并行化方式优化推理性能。



#### 环境准备

| 环境     | 版本           |
| -------- | -------------- |
| Device   | Ascend310/310P |
| CANN     | 6.0.1          |
| mxVision | 3.0.0          |
| Python   | 3.9            |



#### 模型准备

##### 1. paddle转onnx

xxxxxxxxxxxxxxxxxxx

##### 2. onnx转om（模型自动分档）

xxxxxxxxxxxxxxxxxxx



#### 推理

##### 命令示例

- 检测+分类+识别全流程

  ```
  mindocr_infer --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om --cls_model_path=/xxx/cls/cls_310.om --rec_model_path=/xxx/crnn/ --rec_char_dict_path=/xxx/ppocr_keys_v1.txt
  ```

  结果默认保存在inference_results目录下，文件名为pipeline_results.txt

- 检测+识别流程

  不传入--cls_model_path参数，就会跳过方向分类，只执行检测+识别

  ```
  mindocr_infer --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om --rec_model_path=/xxx/crnn/ --rec_char_dict_path=/xxx/ppocr_keys_v1.txt
  ```

  结果默认保存在inference_results目录下，文件名为pipeline_results.txt

- 检测

  可以单独运行文本检测，不传入分类和识别的参数即可

  ```
  mindocr_infer --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om
  ```

  结果默认保存在inference_results目录下，文件名为det_results.txt

- 识别

  可以单独运行文字识别，不传入检测和分类的参数即可

  ```
  mindocr_infer --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om --cls_model_path=/xxx/cls/cls_310.om --rec_model_path=/xxx/crnn/ --rec_char_dict_path=/xxx/ppocr_keys_v1.txt
  ```

  结果默认保存在inference_results目录下，文件名为rec_results.txt

##### 详细参数

| name                   | introduction                                                 | required |
| ---------------------- | ------------------------------------------------------------ | -------- |
| input_images_dir       | 单张图像或者图片文件夹                                       | True     |
| device                 | 推理设备名称（默认Ascend310P3）                              | False    |
| device_id              | 推理设备id（默认0）                                          | False    |
| parallel_num           | 推理流水线中每个节点并行数                                   | False    |
| precision_mode         | 推理的精度模式（暂未实现）                                   | False    |
| det_algorithm          | 文本检测算法名（默认DBNet）                                  | False    |
| rec_algorithm          | 文字识别算法名（默认CRNN)                                    | False    |
| det_model_path         | 文本检测模型的文件路径                                       | False    |
| cls_model_path         | 方向分类模型的文件路径                                       | False    |
| rec_model_path         | 文字识别模型的文件/文件夹路径                                | False    |
| rec_char_dict_path     | 文字识别模型对应的词典文件路径                               | False    |
| res_save_dir           | 推理结果保存的文件夹路径（默认为inference_results），文件名如下<br/>检测+分类+识别/检测+识别pipeline_results.txt：<br/>检测：det_results.txt<br/>识别：rec_results.txt | False    |
| vis_det_save_dir       | 单独的文本检测任务中，结果保存文件夹，保存画有文本检测框的图片 | False    |
| vis_pipeline_save_dir  | 检测+分类+识别/检测+识别的任务中，结果保存文件夹，保存画有文本检测框和文字的图片 | False    |
| vis_font_path          | vis_pipeline_save_dir中绘制图片的字体文件路径（默认采用simfang.ttf） | False    |
| save_pipeline_crop_res | 检测+分类+识别/检测+识别的任务中，是否保存检测结果           | False    |
| pipeline_crop_save_dir | save_pipeline_crop_res为True时，检测结果的文件夹，保存检测后裁剪的图片 | False    |
| show_log               | 是否打印日志（暂未完全支持，默认打印INFO级别日志）           | False    |
| save_log_dir           | 日志保存文件夹（暂未完全支持）                               | False    |

