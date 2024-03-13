English(coming soon) | 中文

# Vary
> [Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](https://arxiv.org/abs/2312.06109)

## 1. 模型描述
Vary是扩展大视觉语言模型（LVLM）视觉词汇的一种有效方法。Vary分为两个部分：新视觉词汇的生成和整合。在第一阶段，Vary设计了一个“词汇网络”以及一个很小的解码器Transformer，并通过自回归产生所需的词汇。然后，Vary通过将新的视觉词汇与原始的视觉词汇（CLIP）合并来扩大普通视觉词汇的规模，使LVLM能够快速获得新的功能。

## 2. 快速开始
### 2.1 环境及数据准备

#### 2.1.1 安装
环境安装教程请参考MindOCR的 [installation instruction](https://github.com/mindspore-lab/mindocr#installation).

#### 2.1.2 配置文件
除了数据集的设置，请同时重点关注以下变量的配置：`seq_length`、`checkpoint_name_or_path`、`repetition_penalty`、`max_decode_length`。说明如下：

```yaml
model:
  ...
  seq_length: 512  # 句子长度
  checkpoint_name_or_path: "/path/to/vary_toy.ckpt"  # 权重路径
  repetition_penalty: 1.1  # 减少生成重复值的惩罚项
  max_decode_length: 512  # 最大生成的句子长度
  ...
...
```

### 2.2 模型推理

```
./tools/infer/text/predict_llm.py --image_dir=/path/to/image.jpg --query="Provide the ocr results of this image."
```

其中image_dir为图片路径，query为查询语句。执行结果将打印到屏幕上。

举例来讲，使用下图和查询语句："Describe this image in within 100 words."的生成结果如下：

![PMC4055390_00006](./images/PMC4055390_00006.jpg)

```txt
system
You should follow the instructions carefully and explain your answers in detail.user
Describe this image in within 100 words.assistant
The article discusses the analysis of traffic signals using deep learning models, specifically focusing on pedestrian crossing data. The authors propose a method to extract features from videos captured by cameras and use them to train a model for predicting pedestrian behavior. They compare their approach with other methods and show that their model outperforms others in terms of accuracy and robustness. The study also highlights the limitations of their approach, such as the need for accurate hand-crafted features and the lack of consideration for different types of vehicles. Overall, the findings suggest the potential of using machine learning models to improve traffic signal analysis and enhance safety.This article is about the use of deep learning models for predicting pedestrian behavior in traffic signals. It compares the performance of different models and highlights the limitations of these approaches.
```

### 2.2 模型训练

coming soon