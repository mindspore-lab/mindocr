# mxOCR

# 1 简介

## 1.1 背景介绍

本参考设计的目的主要包括以下几个方面：

1.为金融、安防、互联网等客户提供基于Atlas做OCR的参考样例，验证可行性；

2.作为客户基于mxBase开发OCR应用的编程样例（下称Demo），开放源代码给客户参考，降低客户准入门槛，包含c++和python两个版本；

3.提供新架构下系统优化案例，打造超高性能原型系统，为客户提供选择Atlas的理由。

本Demo模型选择是面向直排文本，不考虑弯曲文本的情况，并选择JPEG作为输入图像格式，实现识别输入图片中的文字功能。本文档提供对OCR Demo实现方案的说明。

## 1.2 支持的产品

本系统采用Atlas 300I Pro, Atlas 300V,  Atlas 300I作为实验验证的硬件平台。

## 1.3 软件方案介绍

软件方案包含OCR的三个环节：文本检测，方向分类，字符识别。其中文本检测和字符识别是必要环节，方向分类为可选环节。

本Demo支持基于Paddle PP-OCR server 2.0的DBNet(检测)和CRNN(识别)模型 与 Paddle PP-OCR3.0的DBNet(检测)和SVTR(识别)模型进行静态分档推理。

本Demo支持Paddle PP-OCR server 2.0的CRNN模型进行动态Shape推理
（注：Atlas 300I 仅支持静态分档推理）

本Demo的CPP版本支持在310与310P芯片上进行多卡并行推理。

为了提高CPU、NPU资源利用率，实现极致性能，Demo采用了流水并行及多路并行处理方案。

### 代码主要目录介绍

本Demo工程名为mindxsdk-mxocr，根目录下src为源码目录，现将src的子目录介绍如下：
**注意**：代码目录中的cpp/Modules/DbnetPost下的clipper.cpp、clipper.hpp为开源第三方模块，Demo中不包含这两个文件，需用户自行下载这两个文件，然后放在对应位置。
clipper.cpp、clipper.hpp文件下载链接：https://udomain.dl.sourceforge.net/project/polyclipping/clipper_ver6.4.2.zip

```
.
├── src
│   └── demo
│       ├── cpp
│       │   ├── main.cpp          // c++版Demo 主函数
│       │   ├── build.sh          
│       │   ├── CMakeLists.txt
│       │   ├── ascendbase
│       │   │   ├── BlockingQueue  // 阻塞队列模块，用于实现流水并行
│       │   │   ├── CommandParser  // 命令行参数解析模块
│       │   │   ├── ConfigParser   // 配置文件解析模块
│       │   │   └── Framework      // 流水并行框架模块                    
│       │   ├── config         
│       │   │   └── setup.config  // 配置文件
│       │   └── Modules
│       │       ├── DataType        // 流水并行时各模块间共用的数据结构
│       │       ├── CrnnPost        // crnn 后处理模块
│       │       ├── DbnetPost       // DBNet 后处理模块
│       │       │   ├── clipper.cpp
│       │       │   ├── clipper.hpp
│       │       │   ├── DbnetPost.cpp
│       │       │   └── DbnetPost.h
│       │       ├── Processors       // 流水并行处理的各个模块
│       │       │   ├── HandOutProcess     // 图片分发模块
│       │       │   ├── DbnetPreProcess    // dbnet前处理操作模块
│       │       │   ├── DbnetInferProcess  // dbnet推理操作模块
│       │       │   ├── DbnetPostProcess   // dbnet后处理操作模块
│       │       │   ├── ClsPreProcess      // 分类模型前处理操作模块
│       │       │   ├── ClsInferProcess    // 分类模型推理操作模块
│       │       │   ├── ClstPostProcess    // 分类模型后处理操作模块
│       │       │   ├── CrnnPreProcess     // 识别模型前处理操作模块
│       │       │   ├── CrnnInferProcess   // 识别模型推理操作模块
│       │       │   ├── CrnnPostProcess    // 识别模型后处理操作模块
│       │       │   └── CollectProcess     // 推理结果保存模块
│       │       ├── Signal           // 程序终止信号处理模块
│       │       └── Utils            // 公共数据结构模块
│       ├── data
│       │   ├── auto_gear         // 识别模型插入argmax算子脚本
│       │   │   ├── atc_helper.py   // atc转换辅助脚本
│       │   │   ├── auto_gear.py    //自动分档工具脚本
│       │   │   ├── auto_select.py  //识别模型自动选择工具脚本
│       │   │   └── __init__.py
│       │   ├── pdmodel2onnx         // 识别模型插入argmax算子脚本
│       │   ├── models               // 310P onnx模型转换为om模型脚本
│       │   └── models_310           // 310 onnx模型转换为om模型脚本
│       ├── eval_script
│       │   ├── eval_script.py       // 精度测试脚本
│       │   └── requirements.txt     // 精度测试脚本的python三方库依赖文件
│       └── python
│           ├── main.py              // python版Demo 主函数
│           ├── requirements.txt     // python三方库依赖文件
│           ├── config               // python 版Demo参考配置文件
│           └── src
│                ├── data_type      //流水并行模块间传输使用的data class
│                ├── framework      //流水并行框架
│                ├── processors     //流水并行模块实现
│                │   ├── classification
│                │   │   └── cls    //paddle pp-ocr mobile 2.0 相关模块
│                │   ├── common     // 解码，图像分发，mini batch收集 相关模块
│                │   ├── detection
│                │   │   └── dbnet  //dbnet 相关模块
│                │   └── recognition
│                │       └── crnn   //crnn 相关模块
│                └── utils          // Demo使用的相关工具
│  
├── README.md
└── version.info
```

# 2 环境搭建

### 2.1 软件依赖说明

**表2-1** 软件依赖说明

| 依赖软件          | 版本              | 依赖说明             |
| -------------   |-----------------| ------------------ |
| CANN            | 5.1.RC2或6.0.RC1 | 提供基础acl接口      |
| mxVision        | 3.0.RC3         | 提供基础mxBase的能力  |


### 2.2 CANN环境变量设置

```bash
. 安装目录/ascend-toolkit/set_env.sh
```

### 2.3 mxVision环境变量设置

```bash
. 安装目录/mxVision/set_env.sh
```

# 3 模型转换及数据集获取

## 3.1 Demo所用模型下载地址

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


## 3.2 Demo所用测试数据集下载地址
数据集ICDAR-2019 LSVT下载地址：


| 名称               | 下载链接              |
| ----------------- | ---------------  |
| 图片压缩包1      | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz|
| 图片压缩包2      | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz|
| 标注文件     | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json|


图片压缩包名为 train_full_images_0.tar.gz 与 train_full_images_1.tar.gz

标签文件名为 train_full_labels.json

### 3.1.1 数据集准备

#### 3.1.1.1 数据集下载
命令参考
```
wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz
wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz
wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json
```

#### 3.1.1.2 数据集目录创建
创建数据集目录
```
mkdir -p ./icdar2019/images
```
#### 3.1.1.3 解压图片并移动到对应目录
解压图片压缩包并移动图片到对应目录
```
tar -zvxf ./train_full_images_0.tar.gz
tar -zvxf ./train_full_images_1.tar.gz
mv train_full_images_0/* ./icdar2019/images
mv train_full_images_1/* ./icdar2019/images
rm -r train_full_images_0
rm -r train_full_images_1
```
#### 3.1.1.4 标签格式转换
label文件格式转换为ICDAR2015格式, 转换脚本位于src/demo/data/label_format_trans/label_format_trans.py

运行标签格式转换脚本工具需要依赖的三方库如下所示：
**表3-1** label_format_trans.py依赖python三方库

| 名称               | 版本              |
| ----------------- | ---------------  |
| numpy      | =1.22.4|
| tqdm      | =4.64.0|


格式转换脚本参考如下：
```
python ./label_format_trans.py --label_json_path=/xx/xx/train_full_labels.json --output_path=/xx/xx/icdar2019/
```

## 3.2 pdmodel模型转换为onnx模型
- **步骤 1**   将下载好的paddle模型转换成onnx模型。
执行以下命令安装转换工具paddle2onnx
  ```
   pip3 install paddle2onnx==0.9.5
  ```
运行paddle2onnx工具需要依赖的三方库如下所示：

**表3-1** paddle2onnx依赖python三方库

| 名称               | 版本              |
| ----------------- | ---------------  |
| paddlepaddle      | 2.3.0|

**DBNet paddle模型转成onnx模型**

PP-OCR server 2.0版本指令参考如下：
```
paddle2onnx --model_dir ./ch_ppocr_server_v2.0_det_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True --input_shape_dict="{'x':[-1,3,-1,-1]}"
```

Paddle PP-OCR3.0版本指令参考如下：
```
paddle2onnx --model_dir ./ch_PP-OCRv3_det_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_PP-OCRv3_det_infer.onnx --opset_version 11 --enable_onnx_checker True
```

CRNN paddle模型转成onnx模型指令参考如下：
```
paddle2onnx --model_dir ./ch_ppocr_server_v2.0_rec_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_rec_infer.onnx --opset_version 11 --enable_onnx_checker True --input_shape_dict="{'x':[-1,3,32,-1]}"
```

SVTR paddle模型转成onnx模型指令参考如下：
```
paddle2onnx --model_dir ./ch_PP-OCRv3_rec_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_PP-OCRv3_rec_infer.onnx --opset_version 11 --enable_onnx_checker True
```

分类模型转成onnx模型指令参考如下：
```
paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_mobile_v2.0_cls_infer.onnx --opset_version 11 --enable_onnx_checker True
```

## 3.3 识别模型插入ArgMax算子
- **步骤 2**   使用算子插入工具insert_argmax给字符识别模型（CRNN/SVTR）插入argmax算子。

转到data/pdmodel2onnx目录下，执行脚本，此处需要传入两个参数：'model_path'，对应ch_ppocr_server_v2.0_rec_infer.onnx或ch_PP-OCRv3_rec_infer.onnx模型所在路径；'check_output_onnx'，为是否需要针对输出模型做校验，默认值为Ture，可选choices=[True, False]

使用算子插入工具insert_argmax插入argmax算子指令参考：
  ```
   python3 insert_argmax.py --model_path /xx/xx/ch_ppocr_server_v2.0_rec_infer.onnx --check_output_onnx True
  ```
**表3-2** 使用自动算子插入工具插入argmax算子。

|参数名称     | 参数含义                                         |默认值     |  可选值|
| ---------- |----------------------------------------------| ------------|----------|
| model_path | 对应ch_ppocr_server_v2.0_rec_infer.onnx或ch_PP-OCRv3_rec_infer.onnx模型所在路径 |''|''|
|check_output_onnx| 是否需要针对输出模型做校验                                |True|True,False|

转换出来的结果位于'model_path'路径下，命名为'ch_ppocr_server_v2.0_rec_infer_argmax.onnx' 或 'ch_PP-OCRv3_rec_infer_argmax.onnx'的onnx模型文件。

**表3-3** insert_argmax.py脚本依赖python三方库

| 名称          | 版本       |
|-------------|----------|
| onnx        | >=1.9.0  |
| onnxruntime | >=1.13.1 |

## 3.4 静态分档模型转换

****进行静态分档模型转换前需设置CANN环境变量**** 默认的路径为
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```
请以实际安装路径为准

## 3.4.1静态分档工具
- **步骤 3**   使用静态分档工具auto_gear将onnx模型转换成om模型。

转到data/auto_gear文件下，执行auto_gear静态分档转换。

指令参考如下：
```
python auto_gear.py --image_path=/xx/xx/icdar2019/images --gt_path=/xx/xx/icdar2019/labels --det_onnx_path=/xx/xx/ch_ppocr_server_v2.0_det_infer.onnx --rec_onnx_path=/xx/xx/ch_ppocr_server_v2.0_rec_infer_argmax.onnx --rec_model_height=32 --soc_version=Ascend310P3 --output_path=./output
```
其中的参数'image_path'和'gt_path'分别为数据集的图片路径与标签路径，具体路径按照实际情况修改；参数'det_onnx_path'为上一步dbnet转换所得的onnx模型；参数'rec_onnx_path'为上一步crnn转换得到的onnx模型再插入argmax算子所得。

**rec_model_height请进行合理的设置，ch_ppocr_server_v2.0_rec_infer_argmax.onnx对应的值是32, ch_PP-OCRv3_rec_infer_argmax.onnx对应的值是48**

输出在同级目录的output文件夹中，为不同挡位的om模型文件。

**表3-4** 静态分档工具auto_gear脚本运行参数。

|参数名称| 参数含义                               |默认值|
|-----|------------------------------------|-----|
|image_path| 数据集图片所在路径                       |""|
|det_gear_limit| 检测模型限制的档位数量         |100|
|det_limit_side_len| dbnet前处理指定的最长边的长度               |960|
|det_strategy| 检测模型档位生成的策略，请在max_min和mean_std中选择               |"mean_std"|
|det_expand_ratio| 检测模型档位档位的膨胀系数，仅在策略为max_min时生效               |0.2|
|det_n_std| 检测模型档位档位的膨胀系数，仅在策略为mean_std时生效               |3|
|det_width_range| 检测模型档位的宽的范围               |"1,8192"|
|det_height_range| 检测模型档位的高的范围               |"1,8192"|
|gt_path| 数据集标签txt文件所在路径，格式请使用icdar2015格式                    |''|
|rec_gear_limit| 识别模型限制的档位数量         |32|
|rec_model_height| 识别模型的高，当模型为CRNN时推荐为32，当模型为SVTR时推荐为48         |''|
|rec_strategy| 识别模型档位生成的策略，请在max_min和mean_std中选择               |"mean_std"|
|rec_expand_ratio| 识别模型档位档位的膨胀系数，仅在策略为max_min时生效               |0.2|
|rec_n_std| 识别模型档位档位的膨胀系数，仅在策略为mean_std时生效               |3|
|rec_height_range| 识别模型档位的高的范围               |"1,8192"|
|rec_multi_batch| 识别模型是否使用多batch               |True|
|rec_model_channel| 识别模型的channel数，当输入为gbr/rgb图时为3，当输入为灰度图时为1，只支持1或者3               |3|
|det_onnx_path| dbnet转换所得的onnx模型路径                 |''|
|rec_onnx_path| crnn/svtr转换得到的onnx模型再插入argmax算子后路径 |''|
|soc_version| 静态分档模型适配的芯片版本，可选值[Ascend310P3, Ascend310] |Ascend310P3|
|output_path| 转换完成的静态分档模型的输出路径 |./output|

**注意**：det_gear_limit与rec_gear_limit参数限制的是根据既定策略计算出的档位数，实际的档位数会高于此设置的单位数。

- **步骤 3**   使用自动挑选工具auto_select自动挑选识别om模型。

在data/auto_gear目录下，运行auto_gear.py脚本实现自动挑选om模型，参考命令如下：
```
python3 auto_select.py --rec_model_path ./output/crnn
```
**表3-5** auto_select.py脚本运行参数

|参数名称|参数含义|默认值|
|------|------|-----|
|rec_model_path|识别模型经静态分档后的om模型路径|''|
|device_id|使用芯片卡编号|0|

输出的om文件保存在rec_model_path同级目录下的'selected'和'unselected'文件夹中。 **请使用selected路径中的模型文件。**

### 3.4.2 分类模型转换
将onnx模型转换成om模型。转换指令参考src/demo/data/models/cls目录下面的对应的atc转换脚本:
```
bash atc.sh
```

**注： ch_ppocr_server_v2.0 与 ch_PP-OCRv3 均使用此分类模型。**


## 3.5 动态Shape模型转换

### 3.5.1 ch_ppocr_server_v2.0_rec模型动态shape
如需使用动态Shape模型请使用以下路径下的指令进行转换。转换指令参考src/demo/data/models/crnn/目录下面的对应的atc转换脚本:
```
bash atc_dynamic.sh
```

**注：当前仅在Atlas 300I Pro 与 Atlas 300V支持动态shape版本的ch_ppocr_server_v2.0_rec模型**


# 4 C++版demo编译运行

## 4.1 系统方案各子模块功能介绍
表4.1 系统方案各子模块功能：

| 序号 | 子系统            | 功能描述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | 图片分发模块     | 从用户指定的文件夹读取图片，并分发给后续的模块。   |
| 2    | DBNet模型前处理     | 主要负责解码和缩放，在310P芯片上使用dvpp解码，310芯片使用opencv解码。   |
| 3    | DBNet模型推理       | 本模块负责将前处理好的图片输入进检测模型并获得模型推理出的Tensor。 |
| 4    | DBNet模型后处理     | 本模块主要负责将输出的Tensor根据与训练一致的后处理流程将输入图片切割为包含文本的子图。|
| 5    | Cls模型前处理      | 对dbnet后处理之后切割的子图做resize和归一化操作以及分类模型推理时的batch划分    |
| 6    | Cls模型推理        | 本模块负责将前处理好的图片输入进分类模型并获得模型推理出的Tensor。 |
| 7    | Cls模型后处理      | 将模型输出的Tensor根据与训练一致的后处理流程将需要翻转的子图翻转180度。 |
| 8    | Crnn模型前处理      | 对dbnet后处理之后切割的子图做resize和归一化操作以及识别模型推理时的batch划分    |
| 9    | Crnn模型推理        | 本模块负责将前处理好的图片输入进识别模型并获得模型推理出的Tensor。支持动态batch和静态分档的推理。 |
| 10    | Crnn模型后处理      | 将模型输出的Tensor根据字典转换为文字识别的结果。 |
| 11    | 推理结果保存模块     | 保存推理结果，并在推理结束时发送停止信号。   |

## 4.2 配置

运行前需要在 `/src/demo/cpp/config/setup.config` 配置以下信息

配置程序运行的deviceId，deviceType及模型路径等信息
**注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。
配置device
  ```bash
  deviceId  = 0           // 进行推理的device的id, 如需使用多卡进行推理请用逗号隔开device的id, 如 0,1,2,3
  deviceType = 310P       // 310 or 310P
  ```

配置待模型路径
  ```bash
  detModelPath = ./models/dbnet/dbnet_dy_dynamic_shape.om  // DBNet模型路径
  recModelPath = ./models/crnn/static                            // 识别模型路径, 静态分档时仅需输入包含识别模型的文件夹的路径即可, 动态shape时需要输入模型的路径
  clsModelPath = ./models/cls/cls_310P.om                       // 分类模型路径
  ```

配置文本识别模型输入要求的高、宽的步长、字符标签文件
  ```bash
  staticRecModelMode = true // true时采用静态分档, false时采用动态shape 
  recHeight = 32            // 识别模型输入要求的高, 需和crnn转om模型的参数对应  [仅需在使用动态shape模型时配置!]
  recMinWidth = 320         // 识别模型输入宽的最小值, 需和crnn转om模型的参数对应 [仅需在使用动态shape模型时配置!]
  recMaxWidth = 2240        // 识别模型输入宽的最大值, 需和crnn转om模型的参数对应 [仅需在使用动态shape模型时配置!]
  dictPath = ./models/crnn/ppocr_keys_v1.txt  // 识别模型字典文件
  ```

配置识别文字的输出结果路径
  ```bash
  saveInferResult = false   // 是否保存推理结果到文件, 默认不保存, 如果需要, 该值设置为true, 并配置推理结果保存路径
  resultPath = ./result     // 推理结果保存路径
  ```
**注意**：推理结果写文件是追加写的，如果推理结果保存路径中已经存在推理结果文件，推理前请手动删除推理结果文件，如果有需要，提前做好备份。

## 4.3 编译

- **步骤 1**   登录服务器操作后台，安装CANN及mxVision并设置环境变量。

- **步骤 2**   将mxOCR压缩包下载至任意目录，如“/home/HwHiAiUser/mxOCR”，解压。

- **步骤 3**   执行如下命令，构建代码。

  ```
   cd /home/HwHiAiUser/mxOCR/mindxsdk-mxocr/src/demo/cpp;
   bash build.sh
  ```

  *提示：编译完成后会生成可执行文件“main”，存放在“/home/HwHiAiUser/mxOCR/mindxsdk-mxocr/src/demo/cpp/dist/”目录下。*

## 4.4 运行
**注意 C++ Demo 运行时日志打印调用的是mxVison里面的日志模块，mxVison默认打印日志级别为error，如果需要查看info日志，请将配置文件logging.conf中的console_level值设为 0 。**
logging.conf文件路径：mxVison安装目录/mxVision/config/logging.conf

### 输入图像约束

仅支持JPEG格式，图片名格式为前缀+下划线+数字的形式，如xxx_xx.jpg。

### 运行程序
**注意 在模型的档位较多，或者设置并发数过大的情况下，有可能会导致超出device内存。请关注报错信息。**

执行如下命令，启动OCR demo程序。

  ```
  ./dist/main -i /xx/xx/icdar2019/images/ -t 1 -use_cls false -config ./config/setup.config  
  ```

  根据屏幕日志确认是否执行成功。

  识别结果存放在“/home/HwHiAiUser/mxOCR/mindxsdk-mxocr/src/demo/cpp/result”目录下。

*提示：输入./dist/main -h可查看该命令所有信息。运行可使用的参数如表4-2 运行可使用的参数说明所示。*

**表4-2** 运行可使用的参数说明

| 选项            | 意义                                | 默认值                        |
| -------------- | ----------------------------------  | -----------------------------|
| -i             | 输入图片所在的文件夹路径                 | ./data/imgDir               |
| -t             | 运行程序的线程数，请根据环境内存设置合适值。 | 1                           |
| -use_cls       | 是否在检测模型之后使用方向分类模型。        | false                       |
| -config        | 配置文件setup.config的完整路径。        | ./config/setup.config        |

### 结果展示

OCR识别结果保存在配置文件中指定路径的infer_img_x.txt中（x 为图片id）
每个infer_img_x.txt中保存了每个图片文本框四个顶点的坐标位置以及文本内容，格式如下：
  ```bash
  1183,1826,1711,1837,1710,1887,1181,1876,签发机关/Authority
  2214,1608,2821,1625,2820,1676,2212,1659,有效期至/Dateofexpin
  1189,1590,1799,1606,1797,1656,1187,1641,签发日期/Dateofissue
  2238,1508,2805,1528,2802,1600,2235,1580,湖南/HUNAN
  2217,1377,2751,1388,2750,1437,2216,1426,签发地点/Placeofis
  ```
**注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。

# 4.5 动态库依赖说明

Demo动态库依赖可参见代码中“src”目录的“CMakeLists.txt”文件中“target_link_libraries”参数处。

**表4-3** 动态库依赖说明

| 依赖软件             | 说明                                |
| ------------------ | ------------------------------------|
| libascendcl.so     | ACL框架接口，具体介绍可参见ACL接口文档。   |
| libacl_dvpp.so     | ACL框架接口，具体介绍可参见ACL接口文档。   |
| libpthread.so      | C++的线程库。                         |
| libglog.so         | c++日志库。                           |
| libopencv_world.so | OpenCV的基本组件，用于图像的基本处理。     |
| libmxbase.so       | 基础SDK的基本组件，用于模型推理及内存拷贝等。|

# 5 Python版demo编译运行

## 5.1 依赖python三方库安装
Demo运行依赖的三方库如表4-1所示

**表5-1** 依赖python三方库

| 名称               | 版本              |
| ----------------- | ---------------  |
| shapely           | >=1.8.2          |
| pyclipper         | >=1.3.0.post3    |
| textdistance      | >=4.3.0          |
| numpy             | >=1.22.4         |
| mindx             | >=3.0rc3         |

安装命令
  ```
  pip3 install + 包名
  ```

注：mindx的wheel包一般会在安装mxVision的包时自动安装，如果没有成功安装，可以在mxVision的python目录下找到wheel包，并手动pip install。

## 5.2 运行

### 输入图像约束

仅支持JPEG格式，图片名格式为前缀+下划线+数字的形式，如xxx_xx.jpg。

### 修改配置文件

配置文件结构参考
  ```bash
# 流水线中模型的排列顺序，当前只支持 [ 'dbnet', 'crnn' ] 与 [ 'dbnet', 'cls', 'crnn' ] 两种输入
module_order: [ 'dbnet','cls','crnn' ]

# 推理使用的芯片类型，当前只支持'Ascend310P3'与'Ascend310'两种输入，当输入为Ascend310P3时使用DVPP解码，其他情况使用OpenCV解码
device: 'Ascend310P3'

# 全局设置的芯片编号，支持单个数字或列表输入，当输入为列表时将启用多芯片推理，设置前请用npu-smi info确定芯片id与个数
device_id: [ 0 ]

# DBNet相关设置，当流水线模型排列中有dbnet时必须设置相关内容
dbnet:
   model_path: './models/dbnet/dbnet_dy_dynamic_shape.om'  #DBNet模型路径, 输入必须为模型文件的路径

# 分类模型相关设置，当流水线模型排列中有cls时必须设置相关内容
cls:
   model_path: './models/cls/cls_310P.om'    #分类模型的路径，输入必须为模型文件的路径

# CRNN/SVTR模型相关设置，当流水线模型排列中有crnn时必须设置相关内容
# 当static_method为True时，输入可以是模型文件的路径或者是仅包含分档识别模型的文件夹的路径。 当static_method为False，输入仅支持模型文件的路径
# 请使用model_dir输入仅包含分档识别模型的文件夹路径
# 请使用model_path输入识别模型的文件路径
crnn:
   static_method: True  #是否使用分档模型，默认为True，当使用动态shape模型时请设置为False
   model_dir: './models/crnn/static'       # 仅包含识别模型文件夹的路径。
   dict_path: './models/crnn/ppocr_keys_v1.txt'    # 识别模型字典的路径，输入必须为识别模型字典的具体路径。
   device_id: [0] # 指定模型使用device id进行推理，支持单个数字或列表输入。当全局device id与模型指定device id都设置时，优先使用模型指定的device进行推理。默认使用全局device id进行设置。
  model_height: 32  #识别模型输入要求的高, 需和crnn转om模型的参数对应 [仅需在使用动态shape模型时配置!]
  model_min_width: 32 #识别模型输入宽的最小值, 需和crnn转om模型的参数对应 [仅需在使用动态shape模型时配置!]
  model_max_width: 4096 #识别模型输入宽的最大值, 需和crnn转om模型的参数对应 [仅需在使用动态shape模型时配置!]
  ```

### 运行程序
**注意 在模型的档位较多，或者设置并发数过大的情况下，有可能会导致超出device内存。请关注报错信息。**

- **步骤 1**   登录服务器操作后台，安装CANN及mxVision并设置环境变量。
- **步骤 2**   将mxOCR压缩包下载至任意目录，如“/home/HwHiAiUser/mxOCR”，解压。
- **步骤 3**   执行如下命令，启动OCR demo程序，命令中各个参数路径请根据实际情况指定。

  ```
  cd /home/HwHiAiUser/mxOCR/mindxsdk-mxocr/src/demo/python;
  python3 main.py --input_images_path=./input_imgs --config_path=./config/static_ocr_without_cls.yaml 
  --parallel_num=1 --infer_res_save_path=./result
  ```

  根据屏幕日志确认是否执行成功。

  识别结果存放在 --infer_res_save_path 参数指定的目录下。
  
*提示：输入python3 main -h可查看该命令所有信息。运行可使用的参数如表5-2 运行可使用的参数说明所示。*

**表5-2** 运行可使用的参数说明

| 选项                      | 意义                                      | 默认值                        |
| ------------------------ | ---------------------------------------  | -----------------------------|
| --input_images_path      | 输入图片的文件夹路径。                        | 必选项，需用户指定              |
| --config_path        | 配置文件输入路径。                         | 必选项，需用户指定                     |
| --parallel_num           | 并发数，请根据环境内存设置合适值。               | 1                            |
| --infer_res_save_path    | 推理结果保存路径。<br /> 默认不保存推理结果，需指定输出路径后才会保存。***注意：如果输出路径已经存在，将会抹除输出路径重新创建文件夹***   | ""       |

### 结果展示

OCR识别结果保存在配置文件中指定路径的infer_img_x.txt中（x 为图片id）
每个infer_img_x.txt中保存了每个图片文本框四个顶点的坐标位置以及文本内容，格式如下：
  ```bash
  1183,1826,1711,1837,1710,1887,1181,1876,签发机关/Authority
  2214,1608,2821,1625,2820,1676,2212,1659,有效期至/Dateofexpin
  1189,1590,1799,1606,1797,1656,1187,1641,签发日期/Dateofissue
  2238,1508,2805,1528,2802,1600,2235,1580,湖南/HUNAN
  2217,1377,2751,1388,2750,1437,2216,1426,签发地点/Placeofis
  ```
**注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。

# 6 精度测试脚本使用

## 6.1 依赖python三方库安装

精度测试脚本运行依赖的三方库如表5-1所示

**表6-1** 依赖python三方库

| 名称               | 版本              |
| ----------------- | ---------------  |
| shapely           | >=1.8.2          |
| numpy             | >=1.22.4         |
| joblib            | >=1.1.0          |
| tqdm              | >4.64.0          |

安装命令
  ```
  pip3 install + 包名
  ```

## 6.2 运行

### 运行程序

- **步骤 1**   将mxOCR压缩包下载至任意目录，如“/home/HwHiAiUser/mxOCR”，解压。
- **步骤 2**   运行ocr Demo生成推理结果文件。
- **步骤 3**   执行如下命令，启动精度测试脚本，命令中各个参数请根据实际情况指定。

  ```
  cd /home/HwHiAiUser/mxOCR/mindxsdk-mxocr/src/demo/eval_script;
  python3 eval_script.py --gt_path=/xx/xx/icdar2019/labels --pred_path=/xx/xx/result
  ```

  根据屏幕日志确认是否执行成功。

*运行可使用的参数如表5-2 运行可使用的参数说明所示。*

**表6-2** 运行可使用的参数说明

| 选项                      | 意义                   | 默认值                    |
| ------------------------ |----------------------| ------------------------|
| --gt_path                | 测试数据集标注文件路径。         |    ""         |
| --pred_path              | Ocr Demo运行的推理结果存放路径。 |     ""  |
| --parallel_num           | 并行数。                 | 32                      |