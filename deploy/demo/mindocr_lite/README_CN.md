[English](README.md) | 中文

## MindOCR推理 - Demo


### 数据集

数据下载：分别下载[IC15](https://rrc.cvc.uab.es/?ch=4&com=downloads)和[SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)数据集，用于检测和识别。

格式转换：参考[dataset_converters](https://github.com/mindspore-lab/mindocr/tree/main/tools/dataset_converters) 中的脚本和转换格式，将IC5和SVT的测试集转换为检测和识别的格式。

文本检测的标签文件det_gt.txt格式如下：

```
img_478.jpg	[{"transcription": "SPA", "points": [[1136, 36], [1197, 0], [1220, 49], [1145, 96]]}, {...}]
```

文本识别的标签文件rec_gt.txt格式如下：

```
word_421.png   UNDER
word_1657.png  CANDY
word_1814.png  CATHAY
```


### 模型导出

参考[tools/export.py](../../../tools/export.py)，将自己训练好的ckpt文件导出为mindir文件


```shell
# dbnet_resnet50
python tools/export.py --model_name dbnet_resnet50 --ckpt_load_path=dbnet_resnet50.ckpt
# crnn_resnet34
python tools/export.py --model_name crnn_resnet34 --ckpt_load_path=crnn_resnet34.ckpt
```

或者下载已预先转换好的文件，链接如下：

| 任务   | 模型           | 下载链接                                                     |
|------| -------------- | ------------------------------------------------------------ |
| 文本检测 | DBNet_resnet50 | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-db1df47a-7140cd7e.mindir) |
| 文本识别 | CRNN_resnet34  | [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-eb10a0c9.mindir) |


### 环境准备

1. 在Ascend310/310P上确保安装了CANN包等基础运行环境

2. 参考[MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html)官网教程，[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)2.0.0-rc1版本的Lite云侧推理工具包，以及Python接口Wheel包。

   推理工具包安装时解压即可，并注意设置环境变量：

    ```shell
    export LITE_HOME=/your_path_to/mindspore-lite
    export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
    export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
    ```

    Python接口的Wheel包则使用pip安装：

    ```shell
    pip install mindspore_lite-{version}-{python_version}-linux_{arch}.whl
    ```


### 模型转换

1. DBNet，执行转换脚本，生成新的dbnet_resnet50.mindir文件

    ```shell
    converter_lite --saveType=MINDIR --NoFusion=false --fmk=MINDIR --device=Ascend --modelFile=dbnet_resnet50-db1df47a-7140cd7e.mindir --outputFile=dbnet_resnet50 --configFile=dbnet.txt
    ```

   其中，配置文件dbnet.txt的内容如下：

   ```
   [ascend_context]
   input_format=NCHW
   input_shape=x:[1,3,736,1280]
   ```

2. CRNN，执行转换脚本，生成新的crnn_resnet34.mindir文件

    ```shell
    converter_lite --saveType=MINDIR --NoFusion=false --fmk=MINDIR --device=Ascend --modelFile=crnn_resnet34-83f37f07-eb10a0c9.mindir --outputFile=crnn_resnet34 --configFile=crnn.txt
    ```

   其中，配置文件crnn.txt的内容如下：

   ```
   [ascend_context]
   input_format=NCHW
   input_shape=x:[1,3,32,100]
   ```


### 模型推理

```shell
cd cd deploy/demo/mindocr_lite
```

1. 检测

    ```shell
    python infer.py --input_images_dir=dataset/ic15/det/test/ch4_test_images --device=Ascend --device_id=3 --parallel_num=1 --precision_mode=fp16 --det_model_path=mindir/dbnet_resnet50.mindir --engine=lite --res_save_dir=det_ic15
    ```

    检测结果保存在det_ic15/det_results.txt, 格式如下：

    ```
    img_478.jpg	[[1114, 35], [1200, 0], [1234, 52], [1148, 97]], [...]]
    ```

2. 识别

    ```shell
    python infer.py --input_images_dir=dataset/svt/rec/test/cropped_images --device=Ascend --device_id=3 --parallel_num=1  --precision_mode=fp16 --rec_model_path=mindir/crnn_resnet34.mindir --engine=lite --res_save_dir=rec_svt
    ```

    识别结果保存在rec_svt/rec_results.txt，格式如下：

    ```
    word_421.png	"under"
    word_1657.png	"candy"
    word_1814.png	"cathay"
    ```

3. 检测+识别

    ```shell
    python infer.py --input_images_dir=dataset/ic15/det/test/ch4_test_images --device=Ascend --device_id=3 --parallel_num=1  --precision_mode=fp16 --det_model_path=mindir/dbnet_resnet50.mindir --rec_model_path=mindir/crnn_resnet34.mindir --engine=lite --res_save_dir=det_rec_ic15
    ```

    端到端的检测识别结果保存在det_rec_ic15/pipeline_results.txt，格式如下：

    ```
    img_478.jpg	[{"transcription": "spa", "points": [[1114, 35], [1200, 0], [1234, 52], [1148, 97]]}, {...}]
    ```

    推理执行完毕之后，会打印相应的推理性能FPS值。

infer.py推理命令的详细参数列表如下：

| name | introduction                            | required |
| ---------------------- |-----------------------------------------| -------- |
| input_images_dir | 单张图像或者图片文件夹                             | True |
| device | 推理设备名称（默认Ascend）                        | False |
| device_id | 推理设备id（默认0）                             | False |
| engine | 推理引擎（默认lite）                            | False |
| parallel_num | 推理流水线中每个节点并行数（默认1）                    | False |
| precision_mode | 推理的精度模式（默认fp32）                         | False |
| det_algorithm | 文本检测算法名（默认DBNet）                        | False |
| rec_algorithm | 文字识别算法名（默认CRNN)                         | False |
| det_model_path | 文本检测模型的文件路径                             | False |
| cls_model_path | 方向分类模型的文件路径                             | False |
| rec_model_path | 文字识别模型的文件/文件夹路径                         | False |
| rec_char_dict_path | 文字识别模型对应的词典文件路径                         | False |
| res_save_dir | 推理结果保存的文件夹路径（默认为inference_results）      | False |
| vis_det_save_dir | 文本检测任务中，保存画有文本检测框的图片，不设置则不保存   | False |
| vis_pipeline_save_dir | 检测(+分类)+识别的任务中，保存画有文本检测框和文字的图片，不设置则不保存 | False |
| vis_font_path | 文本识别结果绘制图片所采用的的字体文件路径                   | False |
| pipeline_crop_save_dir | 检测(+分类)+识别别的任务中，保存检测后裁剪的图片，不设置则不保存 | False |
| show_log | 是否打印日志（默认False）                        | False |
| save_log_dir | 日志保存文件夹，不设置则不保存                         | False |


### 精度评估

```shell
cd mindocr/deploy/eval_utils
```

运行如下有3个精度评估脚本：

1. 检测结果评估

    ```shell
    python eval_det.py --gt_path=dataset/ic15/det/test/det_gt.txt --pred_path=det_ic15/det_results.txt
    ```

2. 识别结果评估

    ```shell
    python eval_rec.py --gt_path=dataset/svt/rec/test/rec_gt.txt --pred_path=rec_svt/rec_results.txt
    ```

3. 检测+识别结果评估

    ```shell
    python eval_pipeline.py --gt_path=dataset/ic15/det/test/det_gt.txt --pred_path=det_rec_ic15/pipeline_results.txt
    ```


### Benchmark

测试设置：

- Device: Ascend310P
- MindSpore Lite: 2.0.0-rc1
- CPU: Intel(R) Xeon(R) Gold 6148, 2.40GHz, 2x20 physical cores
- parallel_num: 1

在IC15和SVT的测试集上的精度性能测试结果如下：


1. 检测

| Model          | dataset | recall | precision | f-score | fps   |
| -------------- | ------- | ------ | --------- | ------- |-------|
| DBNet_resnet50 | IC15    | 81.94% | 85.66%    | 83.76%  | 17.18 |

2. 识别

| Model         | dataset | acc    | fps    |
| ------------- | ------- | ------ | ------ |
| CRNN_resnet34 | SVT     | 86.09% | 274.67 |
| CRNN_resnet34 | IC15    | 69.67% | 361.09 |

3. 检测+识别

| Model                        | dataset | acc    | fps   |
| ---------------------------- | ------- | ------ |-------|
| DBNet_resnet50+CRNN_resnet34 | IC15    | 55.80% | 16.28 |
