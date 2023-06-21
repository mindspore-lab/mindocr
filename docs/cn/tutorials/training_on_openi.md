## MindOCR 云上训练快速入门

本文主要介绍MindOCR借助OPENI[启智](https://openi.pcl.ac.cn/)平台的训练方法。

### 迁移外部项目

点击启智平台主页面右上角的加号，从下拉菜单中选择迁移外部项目，将MindOCR从github迁移至启智平台。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/1.png"/>
</div>
输入MindOCR的git url: https://github.com/mindspore-lab/mindocr.git 即可进行迁移。


### 准备数据集

可以上传自己的数据集，也可以关联平台已有的数据集。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/3.png"/>
</div>
上传个人数据集需将可用集群选择为NPU.
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/4.png"/>
</div>

### 准备预训练模型(可选)

如需加载预训练权重，可在模型选项卡中添加。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/5.png"/>
</div>
导入本地模型时，模型框架续为MindSpore.
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/6.png"/>
</div>

### 新建训练任务

在云脑选项卡中选择训练任务->新建训练任务。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/7.png"/>
</div>
基本信息中的计算资源选择为Ascend NPU.
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/8.png"/>
</div>
设置参数并添加运行参数。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/12.png"/>
</div>

* 如需加载预训练权重，可在选择模型中选择已上传的模型文件，并在运行参数中增加ckpt_dir参数，参数值为/cache/*.ckpt，*为实际的文件名
* AI引擎中需选择mindspore 1.9或以上的版本，启动文件为`tools/train.py`
* 运行参数需添加`enable_modelarts`，值为True
* 运行参数中由`config`参数指定具体的模型算法，参数值前缀为/home/work/user-job-dir/运行版本号，新建训练任务的运行版本号通常为V0001

<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/13.png"/>
</div>

### 修改已有训练任务

点击已有训练任务的修改按钮，可以基于已有训练任务进行参数修改并运行新的训练任务。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/14.png"/>
</div>

注意：运行版本号=所基于版本号+1
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/15.png"/>
</div>

### 状态查看

点击相应的任务名称，即可查看配置信息、日志、资源占用情况，进行结果下载。
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/10.png"/>
</div>

<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/11.png"/>
</div>


## Reference

[1] Modified from https://github.com/mindspore-lab/mindyolo/blob/master/tutorials/cloud/openi_CN.md
