English | [中文](README_CN.md)

# VisionLAN

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> VisionLAN: [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)

## Introduction

### VisionLAN

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

## Requirements

| mindspore  | ascend driver  |   firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:-------------:|:-------------------:|
|   2.5.0    |    24.1.0      |   7.5.0.3.220  |     8.0.0.beta1    |


## Quick Start

### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### Dataset preparation

**Training sets**

The authors of VisionLAN used two synthetic text datasets for training: SynthText(800k) and MJSynth. Please follow the instructions of the [original VisionLAN repository](https://github.com/wangyuxin87/VisionLAN) to download the training sets.

After download `SynthText.zip` and `MJSynth.zip`, please unzip and place them under `./datasets/train`. The training set contain 14,200,701 samples in total. More details are as follows:


- [SynText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c): 25GB, 6,976,115 samples<br>
- [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/): 21GB, 7,224,586 samples

**Validation sets**

The authors of VisionLAN used six real text datasets for evaluation: IIIT5K Words (IIIT5K_3000) ICDAR 2013 (IC13_857), Street View Text (SVT), ICDAR 2015 (IC15), Street View Text-Perspective (SVTP), CUTE80 (CUTE). We used the sum of the six benchmarks as validation sets. Please follow the instructions of the [original VisionLAN repository](https://github.com/wangyuxin87/VisionLAN) to download the validation sets.

After download `evaluation.zip`, please unzip this zip file, and place them under `./datasets`. Under `./datasets/evaluation`, there are seven folders:


- [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50M, 3000 samples<br>
- [IC13](http://rrc.cvc.uab.es/?ch=2): 72M, 857 samples<br>
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4M, 647 samples<br>
- [IC15](http://rrc.cvc.uab.es/?ch=4): 21M, 1811 samples<br>
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8M, 645 samples<br>
- [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8M, 288 samples<br>
- Sumof6benchmarks: 155M, 7248 samples

During training, we only used the data under `./datasets/evaluation/Sumof6benchmarks` as the validation sets. Users can delete the other folders `./datasets/evaluation` optionally.


**Test Sets**

We choose ten benchmarks as the test sets to evaluate the model's performance. Users can download the test sets from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) (ref: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)). Only the `evaluation.zip` is required for testing.

After downloading the `evaluation.zip`, please unzip it, and rename the folder name from `evaluation` to `test`. Please place this folder under `./datasets/`.

The test sets contain 12,067 samples in total. The detailed information is as follows:


- [CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html): 8.8 MB, 288 samples<br>
- [IC03_860](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 36 MB, 860 samples<br>
- [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions): 4.9 MB, 867 samples<br>
- [IC13_857](http://rrc.cvc.uab.es/?ch=2): 72 MB, 857 samples<br>
- [IC13_1015](http://rrc.cvc.uab.es/?ch=2): 77 MB, 1015 samples<br>
- [IC15_1811](http://rrc.cvc.uab.es/?ch=4): 21 MB, 1811 samples<br>
- [IC15_2077](http://rrc.cvc.uab.es/?ch=4): 25 MB, 2077 samples<br>
- [IIIT5k_3000](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html): 50 MB, 3000 samples<br>
- [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset): 2.4 MB, 647 samples<br>
- [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf): 1.8 MB, 645 samples


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

### Update yaml config file

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
- As the global batch size  (batch_size x num_devices) is important for reproducing the result, please adjust `batch_size` accordingly to keep the global batch size unchanged for a different number of NPUs, or adjust the learning rate linearly to a new global batch size.


### Training

The training stages include Language-free (LF) and Language-aware (LA) process, and in total three steps for training:

```text
LF_1: train backbone and VRM, without training MLM
LF_2: train MLM and finetune backbone and VRM
LA: using the mask generated by MLM to occlude feature maps, train backbone, MLM, and VRM
```

We used distributed training for the three steps. For standalone training, please refer to the [recognition tutorial](../../../docs/en/tutorials/training_recognition_custom_dataset.md#model-training-and-evaluation).

```shell
# worker_num is the total number of Worker processes participating in the distributed task.
# local_worker_num is the number of Worker processes pulled up on the current node.
# The number of processes is equal to the number of NPUs used for training. In the case of single-machine multi-card worker_num and local_worker_num must be the same.
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1.yaml
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2.yaml
msrun --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA.yaml

# Based on verification,binding cores usually results in performance acceleration.Please configure the parameters and run.
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1.yaml
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2.yaml
msrun --bind_core=True --worker_num=4 --local_worker_num=4 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA.yaml
```
**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html).

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_visionlan`.


### Test

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

## Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### Accuracy

According to our experiments, the evaluation results on ten public benchmark datasets is as follow:

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

**Notes:**

- Train datasets: MJ+ST stands for the combination of two synthetic datasets, SynthText(800k) and MJSynth.
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [Dataset preparation](#dataset-preparation) section.
- The input Shape of MindIR of VisionLAN is (1, 3, 64, 256).



## MindSpore Lite Inference

To inference with MindSpot Lite on Ascend 310, please refer to the tutorial [MindOCR Inference](../../../docs/en/inference/inference_tutorial.md). In short, the whole process consists of the following steps:

**Model Export**

Please [download](#2-results) the exported MindIR file first, or refer to the [Model Export](../../../docs/en/inference/convert_tutorial.md#1-model-export) tutorial and use the following command to export the trained ckpt model to  MindIR file:

```shell
# For more parameter usage details, please execute `python tools/export.py -h`
python tools/export.py --model_name_or_config visionlan_resnet45 --data_shape 64 256 --local_ckpt_path /path/to/visionlan-ckpt
```

The `data_shape` is the model input shape of height and width for MindIR file. The shape value of MindIR in the download link can be found in [Notes](#2-results) under results table.


**Environment Installation**

Please refer to [Environment Installation](../../../docs/en/inference/environment.md) tutorial to configure the MindSpore Lite inference environment.

**Model Conversion**

Please refer to [Model Conversion](../../../docs/en/inference/convert_tutorial.md#2-mindspore-lite-mindir-convert),
and use the `converter_lite` tool for offline conversion of the MindIR file.

**Inference**

Assuming that you obtain output.mindir after model conversion, go to the `deploy/py_infer` directory, and use the following command for inference:

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --rec_model_path=your_path_to/output.mindir \
    --rec_model_name_or_config=../../configs/rec/visionlan/visionlan_resnet45_LA.yaml \
    --res_save_dir=results_dir
```


## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang: From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network. ICCV 2021: 14174-14183
