## MindOCR OpenI Training Guideline

This tutorial introduces the training method of MindOCR using the [OPENI](https://openi.pcl.ac.cn/) platform.


### New Migration

Click on the plus sign and choose to New Migration to migrate MindOCR from github to the Openi platform.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/5c526c9e-0979-4bcf-81e4-ae448beab7df">
</div>
Enter the MindOCR git url: https://github.com/mindspore-lab/mindocr.git



### Prepare Dataset

You can upload your own dataset or associate it with existing datasets on the platform.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/e5701090-6bbc-4cbb-9c96-07e6ecf1fcde">
</div>
Uploading personal datasets requires selecting the available cluster as NPU.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/53098b94-88e3-407b-9062-eac155ee6424">
</div>

### Prepare pretrained model (optional)

To load pre training weights, you can add them in the Model tab.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/be3f1237-05c2-40cb-9587-23f3b5f4d64c">
</div>
When importing a local model, the model framework continues to be MindSpore
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/6bebcca2-72ad-4c09-a5ea-537f84065626">
</div>

### New Training Task

Select Training Task ->New Training Task in the Cloudbrain.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/b8081a09-0e99-4c01-b986-860e0cb525f0">
</div>
The computing resource selection in the basic information is Ascend NPU
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/51ef33fa-b03e-4f6c-9dab-786f898816cd">
</div>
Set parameters and add running parameters.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/eb3aa8cf-81f7-4ab5-a0f8-31aa04fce1e8">
</div>

* To load pre training weights, you can select the uploaded model file in the selection model and add ckpt_dir in the run parameters, parameter has a value of/cache/*. ckpt, where * is the actual file name.
* In the AI engine, it is necessary to select mindpoint version 1.9 or higher, and the start file is `tools/train.py`
* To run parameters, add `enable_modelarts` with a value of True.
* The specific model algorithm is specified by the `config` parameter in the running parameters. The prefix of the parameter value is /home/work/user-job-dir/running-version-number. The running-version-number for the newly created training task is usually V0001.


### Modify existing training tasks

Click the modify button of an existing training task to modify parameters based on the existing training task and run a new training task.

<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/2efcd351-dd4a-4d6c-b0bd-0f8f82c12996">
</div>

Note: Running-version-number=Parents Version +1
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/18af7f35-a4e7-4c78-802d-3aa42b9edc78">
</div>

### Status View

Click on the corresponding task name to view configuration information, logs, resource occupancy, and model download the.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/983d4525-45c9-4fa4-96f8-9359dc1322d8">
</div>

<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/b5e10ca2-b4ef-487f-928e-b5e5a8eeacbc">
</div>


## Reference

[1] Modified from https://github.com/mindspore-lab/mindyolo/blob/master/tutorials/cloud/openi.md
