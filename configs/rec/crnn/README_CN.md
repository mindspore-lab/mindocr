[English](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/README.md) | 中文

# CRNN
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://https://arxiv.org/abs/1507.05717)

## 模型描述
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

卷积递归神经网络 (CRNN) 将 CNN 特征提取和 RNN 序列建模以及转录集成到一个统一的框架中。

如架构图（图 1）所示，CRNN 首先通过卷积层从输入图像中提取特征序列。由此一来，图像由提取的序列特征图表示，其中每个向量都与输入图像上的感受野相关联。 为了进一步处理特征，CRNN 采用循环神经网络层来预测每个帧的标签分布。为了将分布映射到文本字段，CRNN 添加了一个转录层，以将每帧预测转换为最终标签序列。 [<a href="#references">1</a>]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/26082447/224601239-a569a1d4-4b29-4fa8-804b-6690cb50caef.PNG" width=450 />
</p>
<p align="center">
  <em> 图1. CRNN架构图 [<a href="#references">1</a>] </em>
</p>

## 评估结果
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

根据我们的实验，在公开基准数据集（IC03，IC13，IC15，IIIT，SVT，SVTP，CUTE）上的评估结果如下：

<div align="center">

| **模型** | **环境配置** |**骨干网络** | **平均准确率**  | **配置文件** | **模型权重下载** | 
|-----------|--------------|------------------|------------|--------------| ------ |
| CRNN (ours)    | D910x8-MS1.8-G | VGG7       | 82.03%         | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_vgg7.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c.ckpt)     |
| CRNN (ours)    | D910x8-MS1.8-G | ResNet34_vd   | 84.45%         | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/crnn/crnn_resnet34.yaml) | [weights](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07.ckpt) |
| CRNN (PaddleOCR) | - | ResNet34_vd | 83.99% | [yaml](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/configs/rec/rec_r34_vd_none_bilstm_ctc.yml) | [weights](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar) |

</div>

#### 注释：
- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。例如，D910x8-MS1.8-G 用于使用图形模式在8张昇腾910 NPU上依赖Mindspore1.8版本进行训练。
- VGG 和 ResNet 模型都是从头开始训练的，无需任何预训练。
- 上述模型是用 MJSynth(MJ)和 SynthText(ST)数据集训练的。更多数据详情，请参考 [数据集准备](#数据集准备)。
- 评估在每个基准数据集上单独测试，平均准确度是所有子数据集的精度平均值。
- PaddleOCR版CRNN，我们直接用的是其[github](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_rec_crnn_en.md)上面提供的已训练好的模型。


## 快速开始
### 环境及数据准备

#### 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 数据集准备
LMDB格式的训练及验证数据集可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (出处: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载。连接中的文件包含多个压缩文件，其中:
- `data_lmdb_release.zip` 包含了完整的一套数据集，有训练集，验证集以及测试集。
- `validation.zip` 是一个整合的验证集。
- `evaluation.zip` 包含多个基准评估数据集。

### 模型训练
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* 分布式训练

使用预定义的训练配置可以轻松重现报告的结果。对于在多个昇腾910设备上的分布式训练，请将配置参数**distribute**修改为**True**，并运行：

```shell
# 在多个 GPU/Ascend 设备上进行分布式训练
mpirun -n 8 python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```
> 如果脚本由 root 用户执行，则必须将 `--allow-run-as-root` 参数添加到 `mpirun` 中。

同样，也可以使用上述`mpirun`命令在多个 GPU 设备上训练模型。

**注意:**  由于全局批大小 （batch_size x num_devices） 是一个重要的超参数，因此建议保持全局批大小不变以进行重现，或将学习率线性调整为新的全局批大小。

* 单卡训练

如果要在没有分布式训练的情况下在较小的数据集上训练或微调模型，请将配置参数 **distribute** 修改为 **False** 并运行：

```shell
# CPU/GPU/Ascend 设备上的单卡训练
python tools/train.py --config configs/rec/crnn/crnn_resnet34.yaml
```

### 模型评估

若要评估已训练模型的准确性，可以使用`eval.py`。请在yaml配置文件的`eval`部分将参数`ckpt_load_path`设置为模型checkpoint的文件路径，然后运行：

```
python tools/eval.py --config configs/rec/crnn/crnn_vgg7.yaml
```

## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. arXiv preprint arXiv:1507.05717, 2015.
