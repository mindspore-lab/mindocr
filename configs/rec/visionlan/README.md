English | [中文](README_CN.md)

# VisionLAN

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> VisionLAN: [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)

## 1. Introduction

### 1.1 VisionLAN

 Visual Language Modeling Network (VisionLAN) [<a href="#5-references">1</a>] is a text recognion model that learns the visual and linguistic information simultaneously via **character-wise occluded feature maps** in the training stage. This model does not require an extra language model to extract linguistic information, since the visual and linguistic information can be learned as a union.

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindocr-asset/main/images/visionlan_architecture.PNG" width=450 />
</p>
<p align="center">
  <em> Figure 1. The architecture of visionlan [<a href="#5-references">1</a>] </em>
</p>



As shown above, the training pipeline of VisionLAN consists of three modules:

- The backbone extract visual feature maps from the input image;

- The Masked Language-aware Module (MLM) takes the visual feature maps and a randomly selected character index as inputs, and generates position-aware character mask map to create character-wise occluded feature maps;

- Finally, the Visual Reasonin Module (VRM) takes occluded feature maps as inputs and makes prediction under the complete word-level supervision.

While in the test stage, MLM is not used. Only the backbone and VRM are used for prediction.

## 2. Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### 2.1 Accuracy

According to our experiments, the evaluation results on ten public benchmark datasets is as follow:

<div align="center">

| **Model** | **Context** | **Backbone**|  **Train Dataset** | **Model Params **|**Avg Accuracy** | **Train Time** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----------: | :--------------: | :----------: | :--------: | :--------: |:----------: |:--------: | :--------: |:----------: |
| visionlan  | D910x4-MS2.0-G | resnet45 | MJ+ST| 42.2M | 90.61%  |  7718s/epoch   | 1,840 | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_1.yaml) | [ckpt files](https://download.mindspore.cn/toolkits/mindocr/visionlan/visionlan_resnet45_ckpts-7d6e9c04.tar.gz) |

</div>

<details open markdown>
  <div align="center">
  <summary>Detailed accuracy results for ten benchmark datasets</summary>

  | **Model** |  **Context** | **IC03_860**| **IC03_867**| **IC13_857**|**IC13_1015** |  **IC15_1811** |**IC15_2077** | **IIIT5k_3000** |  **SVT** | **SVTP** | **CUTE80** | **Average** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |:------: |
  | visionlan | D910x4-MS2.0-G | 96.16% | 95.16%  |  95.92%|   94.19%  | 84.04%  | 77.46%  | 95.53%  | 92.27%  | 85.74%  |89.58% | 90.61%  |

  </div>

</details>

**Notes:**

- Context: Training context denoted as `{device}x{pieces}-{MS version}-{MS mode}`. Mindspore mode can be either `G` (graph mode) or `F` (pynative mode). For example, `D910x4-MS2.0-G` denotes training on 4 pieces of 910 NPUs using graph mode based on MindSpore version 2.0.0.
- Train datasets: MJ+ST stands for the combination of two synthetic datasets, SynthText(800k) and MJSynth.
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [3.2 Dataset preparation](#32-dataset-preparation) section.


## 3. Quick Start

### 3.1 Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### 3.2 Dataset preparation

* Training sets

The authors of VisionLAN used two synthetic text datasets for training: SynthText(800k) and MJSynth. Please follow the instructions of the [original VisionLAN repository](https://github.com/wangyuxin87/VisionLAN) to download the training sets.

After download `SynthText.zip` and `MJSynth.zip`, please unzip and place them under `./datasets/train`. The training set contain 14,200,701 samples in total. More details are as follows:


> [SynText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/): 25GB, 6,976,115 samples<br>
[MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/): 21GB, 7,224,586 samples

* Validation sets

The authors of VisionLAN used six real text datasets for evaluation: IIIT5K Words (IIIT5K_3000) ICDAR 2013 (IC13_857), Street View Text (SVT), ICDAR 2015 (IC15), Street View Text-Perspective (SVTP), CUTE80 (CUTE). We used the sum of the six benchmarks as validation sets. Please follow the instructions of the [original VisionLAN repository](https://github.com/wangyuxin87/VisionLAN) to download the validation sets.

After download `evaluation.zip`, please unzip this zip file, and place them under `./datasets`. Under `./datasets/evaluation`, there are seven folders:


> [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50M, 3000 samples<br>
[IC13](http://rrc.cvc.uab.es/?ch=2): 72M, 857 samples<br>
[SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4M, 647 samples<br>
[IC15](http://rrc.cvc.uab.es/?ch=4): 21M, 1811 samples<br>
[SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8M, 645 samples<br>
[CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8M, 288 samples<br>
Sumof6benchmarks: 155M, 7248 samples

During training, we only used the data under `./datasets/evaluation/Sumof6benchmarks` as the validation sets. Users can delete the other folders `./datasets/evaluation` optionally.


* Test Sets

We choose ten benchmarks as the test sets to evaluate the model's performance. Users can download the test sets from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). Only the `evaluation.zip` is required for testing.

After downloading the `evaluation.zip`, please unzip it, and rename the folder name from `evaluation` to `test`. Please place this folder under `./datasets/`.

The test sets contain 12,067 samples in total. The detailed information is as follows:


> [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples<br>
[IC03_860](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 36 MB, 860 samples<br>
[IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples<br>
[IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples<br>
[IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples<br>
[IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples<br>
[IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples<br>
[IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples<br>
[SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples<br>
[SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples


In the end of preparation, the file structure should be like:

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

### 3.3 Update yaml config file

If the datasets are placed under `./datasets`, there is no need to change the `train.dataset.dataset_root` in the yaml configuration file `configs/rec/visionlan/visionlan_L*.yaml`.

Otherwise, change the following fields accordingly:

```yaml
...
train:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: train                       <--- Update
...
eval:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: evaluation/Sumof6benchmarks <--- Update
...
```

> Optionally, change `train.loader.num_workers` according to the cores of CPU.


Apart from the dataset setting, please also check the following important args: `system.distribute`, `system.val_while_train`, `common.batch_size`. Explanations of these important args:

```yaml
system:
  distribute: True                                                    # `True` for distributed training, `False` for standalone training
  amp_level: 'O0'
  seed: 42
  val_while_train: True                                               # Validate while training
common:
  ...
  batch_size: &batch_size 192                                          # Batch size for training
...
  loader:
      shuffle: False
      batch_size: 64                                                  # Batch size for validation/evaluation
...
```

**Notes:**
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of GPUs/NPUs, or adjust the learning rate linearly to a new global batch size.


### 3.4 Training

The training stages include Language-free (LF) and Language-aware (LA) process, and in total three steps for training:

```text
LF_1: train backbone and VRM, without training MLM
LF_2: train MLM and finetune backbone and VRM
LA: using the mask generated by MLM to occlude feature maps, train backbone, MLM, and VRM
```

We used distributed training for the three steps. For standalone training, please refer to the [recognition tutorial](../../../docs/en/tutorials/training_recognition_custom_dataset.md#model-training-and-evaluation).

```shell
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1.yaml
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2.yaml
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_visionlan`.


### 3.5 Test

After all three steps training, change the `system.distribute` to `False` in `configs/rec/visionlan/visionlan_resnet45_LA.yaml` before testing.

To evaluate the model's accuracy, users can choose from two options:

- Option 1: Repeat the evaluation step for all individual datasets: CUTE80, IC03_860, IC03_867, IC13_857, IC131015, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP. Then take the average score.

An example of evaluation script fort the CUTE80 dataset is shown below.
```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA.yaml"
training_step="LA"

python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=test/CUTE80 eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"

```

- Option 2: Given that all the benchmark datasets folder are under the same directory, e.g. `test/`. And use the script `tools/benchmarking/multi_dataset_eval.py`. The example evaluation script is like:

```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA.yaml"
training_step="LA"

python tools/benchmarking/multi_dataset_eval.py --config $yaml_file --opt eval.dataset.data_dir="test" eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
```


## 4. Inference

Coming Soon...


## 5. References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang: From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network. ICCV 2021: 14174-14183
