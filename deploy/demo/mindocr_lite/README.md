English | [中文](README_CN.md)

## MindOCR Inference - Demo


### Dataset

Download：Please download the [IC15](https://rrc.cvc.uab.es/?ch=4&com=downloads) and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) datasets for text detection and recognition.

Conversion：Please refer to the script and format in the [dataset_converters](https://github.com/mindspore-lab/mindocr/tree/main/tools/dataset_converters), convert the test sets of IC5 and SVT into detection and recognition formats.

The annotation format of det_gt.txt for text recognition follows:

```
img_478.jpg	[{"transcription": "SPA", "points": [[1136, 36], [1197, 0], [1220, 49], [1145, 96]]}, {...}]
```

The annotation format of rec_gt.txt for text recognition follows:

```
word_421.png   UNDER
word_1657.png  CANDY
word_1814.png  CATHAY
```


### Model Export

Please refer to [tools/export.py](../../../tools/export.py), export your trained ckpt to a MindIR file.

```shell
# dbnet_resnet50
python tools/export.py --model_name dbnet_resnet50 --ckpt_load_path=dbnet_resnet50.ckpt
# crnn_resnet34
python tools/export.py --model_name crnn_resnet34 --ckpt_load_path=crnn_resnet34.ckpt
```

Alternatively, download the pre-converted MindIR file with the following link:

| Task             | Model          | Link                                                         |
| ---------------- | -------------- | ------------------------------------------------------------ |
| text detection   | DBNet_resnet50 | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-db1df47a-7140cd7e.mindir) |
| text recognition | CRNN_resnet34  | [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-eb10a0c9.mindir) |


### Environment Installation

1. Ensure the installation of basic runtime environments such as CANN packages on Ascend310/310P.

2. Please refer to the official website tutorial for [MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html), and [download](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html) the version 2.0.0-rc1 cloud-side inference toolkit, as well as the python interface wheel package.

   Just decompress the inference toolkit, and set environment variables:

    ```shell
    export LITE_HOME=/your_path_to/mindspore-lite
    export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
    export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
    ```

   The wheel package of the python interface is installed using pip:

    ```shell
    pip install mindspore_lite-{version}-{python_version}-linux_{arch}.whl
    ```


### Model Conversion

1. DBNet，run the conversion script to generate a new dbnet_resnet50.mindir file.

    ```shell
    converter_lite --saveType=MINDIR --NoFusion=false --fmk=MINDIR --device=Ascend --modelFile=dbnet_resnet50-db1df47a-7140cd7e.mindir --outputFile=dbnet --configFile=dbnet.txt
    ```

    The config file dbnet.txt is as follows:

    ```
    [ascend_context]
    input_format=NCHW
    input_shape=x:[1,3,736,1280]
    ```

2. CRNN，run the conversion script to generate a new crnn_resnet34.mindir file.

    ```shell
    converter_lite --saveType=MINDIR --NoFusion=false --fmk=MINDIR --device=Ascend --modelFile=crnn_resnet34-83f37f07-eb10a0c9.mindir --outputFile=crnn --configFile=crnn.txt
    ```

    The config file crnn.txt is as follows:

   ```
   [ascend_context]
   input_format=NCHW
   input_shape=x:[1,3,32,100]
   ```


### Model Inference

```
cd deploy/demo/mindocr_lite
```

1. Text detection

    ```shell
    python infer.py --input_images_dir=dataset/ic15/det/test/ch4_test_images --device=Ascend --device_id=3 --parallel_num=1 --precision_mode=fp16 --det_model_path=mindir/dbnet_resnet50.mindir --engine=lite --res_save_dir=det_ic15
    ```

    The results will be saved in det_ic15/det_results.txt, with the following format: 
    ```
    img_478.jpg	[[[1114, 35], [1200, 0], [1234, 52], [1148, 97]], [...]]
    ```
   
2. Text recognition

    ```shell
    python infer.py --input_images_dir=dataset/svt/rec/test/cropped_images --device=Ascend --device_id=3 --parallel_num=1  --precision_mode=fp16 --rec_model_path=mindir/crnn_resnet34.mindir --engine=lite --res_save_dir=rec_svt
    ```

    The results will be saved in rec_svt/rec_results.txt, with the following format: 

    ```
    word_421.png	"under"
    word_1657.png	"candy"
    word_1814.png	"cathay"
    ```

3. Text detection + recognition

    ```shell
    python infer.py --input_images_dir=dataset/ic15/det/test/ch4_test_images --device=Ascend --device_id=3 --parallel_num=1  --precision_mode=fp16 --det_model_path=mindir/dbnet_resnet50.mindir --rec_model_path=mindir/crnn_resnet34.mindir --engine=lite --res_save_dir=det_rec_ic15
    ```

    The results will be saved in det_rec_ic15/pipeline_results.txt, with the following format: 
    
    ```
    img_478.jpg	[{"transcription": "spa", "points": [[1114, 35], [1200, 0], [1234, 52], [1148, 97]]}, {...}]
    ```


After the inference is executed, the corresponding inference performance FPS value will be printed.

The detailed parameter list of the infer.py command is as follows:

| name | introduction                            | required |
| ---------------------- |-----------------------------------------| -------- |
| input_images_dir | Input images dir for inference, can be dir containing multiple images or path of single image. | True |
| device | Device type（Default: Ascend）     | False |
| device_id | Device id（Default: 0）     | False |
| engine | Inference engine.（Default: lite） | False |
| parallel_num | Number of parallel in each stage of pipeline parallelism. （Default: 1） | False |
| precision_mode | Precision mode.（Default: fp32） | False |
| det_algorithm | Detection algorithm name.（Default: DBNet） | False |
| rec_algorithm | Recognition algorithm name. (Default: CRNN） | False |
| det_model_path | Detection model file path.   | False |
| cls_model_path | Classification model file path. | False |
| rec_model_path | Recognition model file path or directory which contains multiple recognition models. | False |
| rec_char_dict_path | Character dict file path for recognition models. | False |
| res_save_dir | Saving dir for inference results. (Default: inference_results） | False |
| vis_det_save_dir | Saving dir for visualization of detection results. If it's not set, the results will not be saved. | False |
| vis_pipeline_save_dir | Saving dir for visualization of det+cls(optional)+rec pipeline inference results. If it's not set, the results will not be saved. | False |
| vis_font_path | Font file path for recognition model. | False |
| pipeline_crop_save_dir | Saving dir for images cropped during pipeline. If it's not set, the results will not be saved. | False |
| show_log | Whether show log when inferring. | False |
| save_log_dir | Log saving dir.                  | False |


### Evaluation

```
cd mindocr/deploy/eval_utils
```

Run the following 3 evaluation script:

1. Text detection

    ```shell
    python eval_det.py --gt_path=dataset/ic15/det/test/det_gt.txt --pred_path=det_ic15/det_results.txt
    ```

2. Text recognition

    ```shell
    python eval_rec.py --gt_path=dataset/svt/rec/test/rec_gt.txt --pred_path=rec_svt/rec_results.txt
    ```

3. Text detection + recognition

    ```shell
    python eval_pipeline.py --gt_path=dataset/ic15/det/test/det_gt.txt --pred_path=det_rec_ic15/pipeline_results.txt
    ```


### Benchmark

Test settings：

- Device: Ascend310P
- MindSpore Lite: 2.0.0-rc1
- CPU: Intel(R) Xeon(R) Gold 6148, 2.40GHz, 2x20 physical cores
- parallel_num: 1

The accuracy and performance test results on the IC15 and SVT test sets are as follows:

1. Text detection

| Model          | dataset | recall | precision | f-score | fps   |
| -------------- | ------- | ------ | --------- | ------- |-------|
| DBNet_resnet50 | IC15    | 81.94% | 85.66%    | 83.76%  | 17.18 |

2. Text recognition

| Model         | dataset | acc    | fps    |
| ------------- | ------- | ------ | ------ |
| CRNN_resnet34 | SVT     | 86.09% | 274.67 |
| CRNN_resnet34 | IC15    | 69.67% | 361.09 |

3. Text detection + recognition

| Model                        | dataset | acc    | fps   |
| ---------------------------- | ------- | ------ |-------|
| DBNet_resnet50+CRNN_resnet34 | IC15    | 55.80% | 16.28 |
