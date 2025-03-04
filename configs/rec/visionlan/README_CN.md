[English](README.md) | 中文

# VisionLAN

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> VisionLAN: [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)

## 简介

### VisionLAN

视觉语言建模网络（VisionLAN）[<a href="#5-引用文献">1</a>]是一种文本识别模型，它通过在训练阶段使用逐字符遮挡的特征图来同时学习视觉和语言信息。这种模型不需要额外的语言模型来提取语言信息，因为视觉和语言信息可以作为一个整体来学习。
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindocr-asset/main/images/visionlan_architecture.PNG" width=450 />
</p>
<p align="center">
  <em> 图 1. Visionlan 的模型结构 [<a href="#5-引用文献">1</a>] </em>
</p>

如上图所示，VisionLAN的训练流程由三个模块组成：

- 骨干网络从输入图像中提取视觉特征图；
- 掩码语言感知模块（MLM）以视觉特征图和一个随机选择的字符索引作为输入，并生成位置感知的字符掩码图，以创建逐字符遮挡的特征图；
- 最后，视觉推理模块（VRM）以遮挡的特征图作为输入，并在完整的单词级别的监督下进行预测。

但在测试阶段，MLM不被使用。只有骨干网络和VRM被用于预测。

## 配套版本

| mindspore  | ascend driver  |   firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:-------------:|:-------------------:|
|   2.5.0    |    24.1.0      |   7.5.0.3.220  |     8.0.0.beta1    |

## 快速入门

### 安装

请参考[MindOCR中的安装说明](https://github.com/mindspore-lab/mindocr#installation)。

### 数据准备

**训练集**

VisionLAN的作者使用了两个合成文本数据集进行训练：SynthText（800k）和MJSynth。请按照[原始VisionLAN repository](https://github.com/wangyuxin87/VisionLAN)的说明进行操作，下载这两个LMDB数据集。

下载`SynthText.zip`和`MJSynth.zip`后，请解压缩并将它们放置在`./datasets/train`目录下。训练集数据总共包括 14,200,701 个样本。更多关于训练集的信息如下:

- [SynText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c): 25GB, 6,976,115 samples
- [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/): 21GB, 7,224,586 samples



**验证集**

VisionLAN的作者使用了六个真实文本数据集进行评估：IIIT5K Words（IIIT5K_3000）、ICDAR 2013（IC13_857）、Street View Text（SVT）、ICDAR 2015（IC15_1811）、Street View Text-Perspective（SVTP）、CUTE80（CUTE）。

请按照[原始VisionLAN repository](https://github.com/wangyuxin87/VisionLAN)的说明进行操作，下载验证数据集。

下载`evaluation.zip`后，请解压缩并将其放置在`./datasets`目录下。在 `./datasets/evaluation`路径下，一共有7个文件夹:

- [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50M, 3000 samples<br>
- [IC13](http://rrc.cvc.uab.es/?ch=2): 72M, 857 samples<br>
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4M, 647 samples<br>
- [IC15](http://rrc.cvc.uab.es/?ch=4): 21M, 1811 samples<br>
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8M, 645 samples<br>
- [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8M, 288 samples<br>
- Sumof6benchmarks: 155M, 7248 samples


训练中，我们只用到了 `./datasets/evaluation/Sumof6benchmarks` 作为验证集。 用户可以选择将 `./datasets/evaluation` 下其他无关的文件夹删除。

**测试集**

我们选择用10个公开数据集来测试模型精度。用户可以从[这里](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here))下载测试集。测试中我们只需要用到 `evaluation.zip`。

在下载 `evaluation.zip`完成后, 请解压缩, 并把文件夹名称从 `evaluation` 改为 `test`。请把这个`test`文件夹放在 `./datasets/`路径下

测试集总共包含 12,067 个样本。详细信息如下：

- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples<br>
- [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples<br>
- [IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples<br>
- [IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples<br>
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples<br>
- [IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples<br>
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples<br>
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples<br>
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples


准备好的数据集文件结构应如下所示：
``` text
datasets
├── test
│   ├── CUTE80
│   ├── IC03_860
│   ├── IC03_867
│   ├── IC13_857
│   ├── IC13_1015
│   ├── IC15_1811
│   ├── IC15_2077
│   ├── IIIT5k_3000
│   ├── SVT
│   ├── SVTP
├── evaluation
│   ├── Sumof6benchmarks
│   ├── ...
└── train
    ├── MJSynth
    └── SynText
```

### 配置说明

如果数据集放置在`./datasets`目录下，则无需更改yaml配置文件`configs/rec/visionlan/visionlan_L*.yaml`中的`train.dataset.dataset_root`。
否则，请相应地更改以下字段：


```yaml
...
train:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: train                       <--- 更新
...
eval:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: evaluation/Sumof6benchmarks <--- 更新
...
```

> 您也可以选择根据CPU的线程数量来修改 `train.loader.num_workers`.

除了数据集设定以外, 请检查下列的重要参数: `system.distribute`, `system.val_while_train`, `common.batch_size`。这些参数的含义解释如下:


```yaml
system:
  distribute: True                                                    # 分布式训练选择`True`， 单卡训练选择 `False`
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                               # 训练途中进行验证
common:
  ...
  batch_size: &batch_size 192                                          # 训练batch size
...
  loader:
      shuffle: False
      batch_size: 64                                                  # 验证/测试 batch size
...
```

**注意：**
- 由于全局批大小 （batch_size x num_devices） 是对结果复现很重要，因此当NPU卡数发生变化时，调整batch_size以保持全局批大小不变，或将学习率线性调整为新的全局批大小。

### 训练

训练阶段包括无语言（LF）和有语言（LA）过程，总共有三个训练步骤：

```text
LF_1：训练骨干网络和VRM，不训练MLM
LF_2：训练MLM并微调骨干网络和VRM
LA：使用MLM生成的掩码遮挡特征图，训练骨干网络、MLM和VRM
```

我们接下来使用分布式训练进行这三个步骤。对于单卡训练，请参考[识别教程](../../../docs/zh/tutorials/training_recognition_custom_dataset.md#单卡训练)。

```shell
# worker_num代表分布式总进程数量。
# local_worker_num代表当前节点进程数量。
# 进程数量即为训练使用的NPU的数量，单机多卡情况下worker_num和local_worker_num需保持一致。
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1.yaml
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2.yaml
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA.yaml

# 经验证，绑核在大部分情况下有性能加速，请配置参数并运行
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1.yaml
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2.yaml
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA.yaml
```
**注意:** 有关 msrun 配置的更多信息，请参考[此处](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html).

训练结果（包括checkpoints、每个阶段的性能和loss曲线）将保存在yaml配置文件中由参数`ckpt_save_dir`解析的目录中。默认目录为`./tmp_visionlan`。


### 测试

在完成上述三个训练步骤以后, 用户需要在测试前，将 `configs/rec/visionlan/visionlan_resnet45_LA.yaml` 文件中的`system.distribute`改为 `False`。

若要评估已训练模型的准确性, 有以下两个方法可供选择：


- 方法一: 先对每一个数据集进行评估： CUTE80, IC03_860, IC03_867, IC13_857, IC131015, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP。 然后再计算平均精度。

CUTE80 数据集的评估脚本如下：
```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA.yaml"
training_step="LA"

python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=test/CUTE80 eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"

```

- 方法二: 先将全部的测试集放在同一个文件夹下，例如 `datasets/test/`, 然后用 `tools/benchmarking/multi_dataset_eval.py`进行多数据集评估。示例脚本如下:

```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA.yaml"
training_step="LA"

python tools/benchmarking/multi_dataset_eval.py --config $yaml_file --opt eval.dataset.data_dir="test" eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
```

## 精度结果

根据我们实验结果，在10个公开数据集上的评估结果如下：

<div align="center">

| **model name** | **backbone** | **train dataset** | **params(M)** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **accuracy** |                                                                                                                                                                        **recipe**                                                                                                                                                                        |                                                                                                            **weight**                                                                                                             |
|:--------------:|:------------:|:-----------------:|:-------------:|:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   visionlan    |   Resnet45   |      MJ+ST        |     42.22     |     4     |      128       |      O2       |     191.52 s      |   280.29    |  1826.63  |    90.62%    | [yaml(LF_1)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_1.yaml) [yaml(LF_2)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_2.yaml) [yaml(LA)](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LA.yaml)  | [ckpt files](https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_ckpts-7d6e9c04.tar.gz) \| [mindir(LA)](https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_LA-e9720d9e-71b38d2d.mindir)|

</div>

<details open markdown>
  <div align="center">
  <summary>Detailed accuracy results for ten benchmark datasets</summary>

| **model name** | **backbone** | **cards** | **IC03_860** | **IC03_867** | **IC13_857** | **IC13_1015** | **IC15_1811** | **IC15_2077** | **IIIT5k_3000** | **SVT** | **SVTP** | **CUTE80** | **average** |
|:--------------:|:------------:|:---------:|:------------:|:------------:|:------------:|:-------------:|:-------------:|:-------------:|:---------------:|:-------:|:--------:|:----------:|:-----------:|
|   visionlan    |  Resnet45    |     1     |    96.16%    |    95.16%    |    95.92%    |    94.19%     |    84.04%     |    77.47%     |     95.53%      | 92.27%  |  85.89%  |   89.58%   |   90.62%    |

  </div>

</details>

**注**

- 训练数据集：`MJ+ST`代表两个合成数据集SynthText（800k）和MJSynth的组合。
- 要在其他训练环境中重现结果，请确保全局批量大小相同。
- 这些模型是从头开始训练的，没有任何预训练。有关训练和评估的更多数据集详细信息，请参阅[数据准备](#数据准备)部分。
- VisionLAN的MindIR导出时的输入Shape均为(1, 3, 64, 256)。


## MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/zh/inference/inference_tutorial.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

**模型导出**

请先[下载](#2-评估结果)已导出的MindIR文件，或者参考[模型导出](../../../docs/zh/inference/convert_tutorial.md#1-模型导出)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
# 有关更多参数使用详细信息，请执行 `python tools/export.py -h`
python tools/export.py --model_name_or_config visionlan_resnet45 --data_shape 64 256 --local_ckpt_path /path/to/visionlan-ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[注释](#2-评估结果)。

**环境搭建**

请参考[环境安装](../../../docs/zh/inference/environment.md)教程，配置MindSpore Lite推理运行环境。

**模型转换**

请参考[模型转换](../../../docs/zh/inference/convert_tutorial.md#2-mindspore-lite-mindir-转换)教程，使用`converter_lite`工具对MindIR模型进行离线转换。


**执行推理**

假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/visionlan/visionlan_resnet45_LA.yaml \
    --res_save_dir=results_dir
```

## 引用文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang: From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network. ICCV 2021: 14174-14183
