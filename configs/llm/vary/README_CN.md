[English](README.md) | 中文

# Vary-toy
> [Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](https://arxiv.org/abs/2312.06109)
> [Small Language Model Meets with Reinforced Vision Vocabulary](https://arxiv.org/abs/2401.12503)

## 1. 模型描述
Vary是扩展大视觉语言模型（LVLM）视觉词汇的一种有效方法。Vary分为两个部分：新视觉词汇的生成和整合。在第一阶段，Vary设计了一个“词汇网络”以及一个很小的解码器Transformer，并通过自回归产生所需的词汇。然后，Vary通过将新的视觉词汇与原始的视觉词汇（CLIP）合并来扩大普通视觉词汇的规模，使LVLM能够快速获得新的功能。Vary-toy是Vary官方开源的较小规模版本。

## 2. 评估结果

根据我们的实验，Vary-toy的推理性能如下：

<div align="center">

| **模型**  | **环境配置**    | **总时间**     | **token生成速度**     | **配置文件** | **模型权重下载** |
| :-----:  | :--------:     | :--------:     | :-----:     | :---------: | :---------: |
| Vary-toy | D910x1-MS2.2-G | 23.38 s | 30.75 tokens/s | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/llm/vary/vary_toy.yaml)| [ckpt](https://download-mindspore.osinfra.cn/toolkits/mindocr/vary/vary_toy-e62a3564.ckpt) |
</div>

**注意:**

- 环境配置：训练的环境配置表示为 {处理器}x{处理器数量}-{MS模式}，其中 Mindspore 模式可以是 G-graph 模式或 F-pynative 模式。例如，D910x1-MS2.2-G 使用图模式在1张昇腾910 NPU上依赖Mindspore2.2版本进行训练。
- 如需在其他环境配置重现训练结果，请确保全局批量大小与原配置文件保持一致。

## 3. 快速开始
### 3.1 环境及模型准备

#### 3.1.1 安装

注：若你想实验Vary-toy，你要将python升级到3.8或以上版本。

环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation) 。

此外，还需要使用如下shell命令安装`tiktoken`：

```shell
pip install tiktoken
```

#### 3.1.2 配置文件
请重点关注以下变量的配置：`seq_length`、`checkpoint_name_or_path`、`repetition_penalty`、`max_decode_length`、`max_new_tokens`、`vocab_file`。说明如下：

```yaml
model:
  ...
  seq_length: 2048  # 句子长度
  checkpoint_name_or_path: "/path/to/vary_toy.ckpt"  # 权重路径
  repetition_penalty: 1.5  # 生成重复值的惩罚项
  max_decode_length: 2048  # 最大生成的句子长度
  max_new_tokens: 1024  # 生成的新token的个数
  ...
...
processor:
  ...
  tokenizer:
    vocab_file: "/path/to/qwen.tiktoken"  # 分词器路径
  ...
...
```

#### 3.1.3 模型准备

用户可以从下方链接下载分词器模型：

- [qwen.tiktoken](https://huggingface.co/HaoranWei/Vary-toy/blob/main/qwen.tiktoken)

用户可以从下方链接下载权重：

- [Vary-toy](https://download-mindspore.osinfra.cn/toolkits/mindocr/vary/vary_toy-e62a3564.ckpt)

用户也可以从下方huggingface链接下载权重：

- [Vary-toy](https://huggingface.co/HaoranWei/Vary-toy/blob/main/pytorch_model.bin)

然后根据以下步骤进行权重转换：

注：启动转换脚本前请安装`torch`：

```shell
pip install torch
```

下载完成后，运行mindocr/models/llm/convert_weight.py转换脚本，将huggingface的权重转换为MindSpore的ckpt权重。

```shell
python mindocr/models/llm/convert_weight.py \
    --torch_ckpt_path="/path/to/pytorch_model.bin" \
    --mindspore_ckpt_path="/path/to/vary_toy.ckpt"

# 参数说明：
# torch_ckpt_path：huggingface下载的权重路径
# mindspore_ckpt_path：导出的MindSpore权重路径
```

### 3.2 模型推理

```shell
python ./tools/infer/text/predict_llm.py \
    --image_dir=/path/to/image.jpg \
    --query="Provide the ocr results of this image." \
    --config_path="/path/to/vary_toy.yaml" \
    --chat_mode=False

# 参数说明：
# image_dir：图片路径
# query：查询语句
# config_path：配置文件路径
# chat_mode：是否使用对话模式
```

执行结果将打印到屏幕上。

例如，可使用查询语句："Describe this image in within 100 words."，生成对下图文本的分析结果：

![PMC4055390_00006](./images/PMC4055390_00006.jpg)

```txt
The article discusses the analysis of traffic signals using deep learning models, specifically focusing on pedestrian crossing data. The authors propose a method to extract features from videos captured by cameras and use them to train a model for predicting pedestrian behavior. They compare their approach with other methods and show that their model outperforms others in terms of accuracy and robustness. The study also highlights the limitations of their approach, such as the need for accurate hand-crafted features and the lack of consideration for different types of vehicles. Overall, the findings suggest the potential of using machine learning models to improve traffic signal analysis and enhance safety.This article is about the use of deep learning models for predicting pedestrian behavior in traffic signals. It compares the performance of different models and highlights the limitations of these approaches.
```

### 3.3 模型训练

coming soon
