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

#### 模型下载

##### 所用模型下载地址

Paddle PP-OCR server 2.0模型:

| 名称               | 下载链接              |
| ----------------- | ---------------  |
| Paddle PP-OCR server 2.0 DBNet      | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar|
| Paddle PP-OCR server 2.0 Cls      | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar|
| Paddle PP-OCR server 2.0 CRNN      | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar|


Paddle PP-OCR 3.0模型:


| 名称               | 下载链接              |
| ----------------- | ---------------  |
| Paddle PP-OCR3.0 DBNet      | https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar|
| Paddle PP-OCR3.0 Cls      | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar|
| Paddle PP-OCR3.0 SVTR     | https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar|



识别模型字典文件下载地址：
https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/ppocr/utils/ppocr_keys_v1.txt


**注： ch_ppocr_server_v2.0 与 ch_PP-OCRv3 均使用此名为ch_ppocr_mobile_v2.0_cls_infer的分类模型与名为ppocr_keys_v1.txt的识别模型的字典。**


##### 所用测试数据集下载地址
数据集ICDAR-2019 LSVT下载地址：


| 名称               | 下载链接              |
| ----------------- | ---------------  |
| 图片压缩包1      | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz|
| 图片压缩包2      | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz|
| 标注文件     | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json|


图片压缩包名为 train_full_images_0.tar.gz 与 train_full_images_1.tar.gz

标签文件名为 train_full_labels.json

##### 1 数据集准备

数据集准备参考deploy/data_utils/preprocess目录下对应数据的标签格式化转换脚本，并按照脚本里注释的步骤依次下载数据集、新建文件夹、解压文件并执行脚本。

##### 2 模型转换

##### 模型准备

###### 2.1 paddle转onnx

将下载好的paddle模型转换成onnx模型。

运行paddle2onnx工具需要依赖的三方库如下所示：

| 名称               | 版本              |
| ----------------- | ---------------  |
| paddlepaddle      | 2.3.0|
| paddle2onnx       | 0.9.5|

**DBNet paddle模型转成onnx模型**

PP-OCR server 2.0版本指令参考如下：

  ```
  paddle2onnx --model_dir ./ch_ppocr_server_v2.0_det_infer/ --model_filename inference.pdmodel \
              --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_det_infer.onnx \
              --opset_version 11 --enable_onnx_checker True --input_shape_dict="{'x':[-1,3,-1,-1]}"
  ```

Paddle PP-OCR3.0版本指令参考如下：

  ```
  paddle2onnx --model_dir ./ch_PP-OCRv3_det_infer/ --model_filename inference.pdmodel \
              --params_filename inference.pdiparams --save_file ./ch_PP-OCRv3_det_infer.onnx \
              --opset_version 11 --enable_onnx_checker True
  ```

CRNN paddle模型转成onnx模型指令参考如下：

  ```
  paddle2onnx --model_dir ./ch_ppocr_server_v2.0_rec_infer/ --model_filename inference.pdmodel \
              --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_rec_infer.onnx \
              --opset_version 11 --enable_onnx_checker True --input_shape_dict="{'x':[-1,3,32,-1]}"
  ```

SVTR paddle模型转成onnx模型指令参考如下：

  ```
  paddle2onnx --model_dir ./ch_PP-OCRv3_rec_infer/ --model_filename inference.pdmodel \
              --params_filename inference.pdiparams --save_file ./ch_PP-OCRv3_rec_infer.onnx \
              --opset_version 11 --enable_onnx_checker True
  ```

分类模型转成onnx模型指令参考如下：

  ```
  paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel \
              --params_filename inference.pdiparams --save_file ./ch_ppocr_mobile_v2.0_cls_infer.onnx \
              --opset_version 11 --enable_onnx_checker True
  ```

###### 2.2 onnx转om（模型自动分档）

将onnx模型转化为om模型。

参考deploy/models_utils/conversion/onnx2om.sh脚本执行自动串行执行脚本将onnx模型转om模型。

需要适配脚本对应数据和模型参数：

| 参数名称             | 描述                               |
|------------------|----------------------------------|
| model_path       | 需要插入argmax的模型文件路径                |
| image_path       | 数据集图片数据路径                        |
| gt_path          | 数据集标签路径                          |
| det_onnx_path    | det onnx模型路径                     |
| rec_onnx_path    | rec onnx模型路径                     |
| soc_version      | Ascend卡的类型，可选Ascend310、Ascend310P3 |
| rec_model_height | 识别模型的height值，CRNN为32，SVTR为48     |
  ```
  bash onnx2om.sh
  ```

针对Cls的分档

分类模型没有对HW分档，只对N进行分档，不需要在数据集上统计，参考deploy/models_utils/conversion/auto_gear/atc.sh执行转换。

  ```
  bash atc.sh
  ```

### onnx2om.sh脚本包括以下步骤，不需要单独执行，已经集成在脚本中

###### 识别模型插入ArgMax算子

转到deploy/models_utils/conversion/onnx_optim目录下，使用算子插入工具insert_argmax，在文字识别模型（CRNN/SVTR）中插入argmax算子：

  ```
   python3 insert_argmax.py --model_path /xx/xx/ch_ppocr_server_v2.0_rec_infer.onnx --check_output_onnx True
   python3 insert_argmax.py --model_path /xx/xx/ch_PP-OCRv3_rec_infer.onnx --check_output_onnx True
  ```

转换出来的结果位于'model_path'路径下，命名为'ch_ppocr_server_v2.0_rec_infer_argmax.onnx' 或 'ch_PP-OCRv3_rec_infer_argmax.onnx'的onnx模型文件。

###### onnx模型转om模型

这里实现了Shape分档功能。例如，CRNN模型的输入Shape为(N, 3, 32, W)，在模型转换时，N和W设置了多种可选的组合，即为Shape分档。

模型分档时，对于如何设置HW的组合，这里提供了一些自动化脚本，可以从数据集中自动统计，实现自动分档功能。

deploy/models_utils/conversion/auto_gear/auto_gear.py提供了自动分档功能，它基于数据集统计分档参数，然后自动调用ATC工具，实现模型分档与转换。auto_gear.py有很多可选参数，详情见README，本文这里只使用默认参数。

deploy/models_utils/conversion/auto_gear目录，提供了ATC工具的例子，用户可以手动调用。

###### 分档

（1）v2.0的DBNet/CRNN分档

  ```
  python3 auto_gear.py --image_path=/xx/xx/lsvt/images --gt_path=/xx/xx/lsvt/labels --det_onnx_path=/xx/xx/ch_ppocr_server_v2.0_det_infer.onnx --rec_onnx_path=/xx/xx/ch_ppocr_server_v2.0_rec_infer_argmax.onnx --rec_model_height=32 --soc_version=Ascend310P3 --output_path=./lsvt_om_v2
  ```

其中，CRNN模型的H为32，所以rec_model_height设置为32。运行结束后会在output_path目录下生成crnn和dbnet文件夹，crnn下会有多个om文件，dbnet文件夹下只有1个om文件。

（2）v3.0的DBNet/SVTR分档

  ```
  python3 auto_gear.py --image_path=/xx/xx/lsvt/images --gt_path=/xx/xx/lsvt/labels --det_onnx_path=/xx/xx/ch_PP-OCRv3_det_infer.onnx --rec_onnx_path=/xx/xx/ch_PP-OCRv3_rec_infer_argmax.onnx --rec_model_height=48 --soc_version=Ascend310P3 --output_path=./lsvt_om_v3
  ```

其中，SVTR模型的H为48，所以rec_model_height设置为48。运行结束后会在output_path目录下生成svtr和dbnet文件夹，svtr下会有多个om文件，dbnet文件夹下只有1个om文件。

###### 自动选择

SVTR和CRNN在自动分档时会产生多个模型文件，使用自动挑选工具auto_select自动挑选识别性能更优的om模型。

在deploy/models_utils/conversion/auto_gear目录下，参考命令如下：

  ```
  python3 auto_select.py --rec_model_path lsvt_om_v2/crnn
  python3 auto_select.py --rec_model_path lsvt_om_v3/svtr
  ```

完成挑选后，被选中的om文件存在rec_model_path下的selected文件夹下面，后续推理时选择该文件下的模型使用即可。

#### 3 推理

##### 命令示例

- 检测+分类+识别全流程

  ```
  mindocr --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om --cls_model_path=/xxx/cls/cls_310.om --rec_model_path=/xxx/crnn/ --rec_char_dict_path=/xxx/ppocr_keys_v1.txt
  ```

  结果默认保存在inference_results目录下，文件名为pipeline_results.txt

- 检测+识别流程

  不传入--cls_model_path参数，就会跳过方向分类，只执行检测+识别

  ```
  mindocr --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om --rec_model_path=/xxx/crnn/ --rec_char_dict_path=/xxx/ppocr_keys_v1.txt
  ```

  结果默认保存在inference_results目录下，文件名为pipeline_results.txt

- 检测

  可以单独运行文本检测，不传入分类和识别的参数即可

  ```
  mindocr --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om
  ```

  结果默认保存在inference_results目录下，文件名为det_results.txt

- 识别

  可以单独运行文字识别，不传入检测和分类的参数即可

  ```
  mindocr --input_images_dir=/xxx/images --device=Ascend310 --det_model_path=/xxx/dbnet/dbnet_dynamic_dims_100.om --cls_model_path=/xxx/cls/cls_310.om --rec_model_path=/xxx/crnn/ --rec_char_dict_path=/xxx/ppocr_keys_v1.txt
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

