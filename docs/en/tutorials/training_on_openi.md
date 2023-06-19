## MindOCR OpenI Training Guideline

This tutorial introduces the training method of MindOCR using the [OpenI](https://openi.pcl.ac.cn/) platform.


### Clone the project

Click on the plus sign and choose to New Migration to clone MindOCR from GitHub to the Openi platform.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/5c526c9e-0979-4bcf-81e4-ae448beab7df">
</div>
Enter the MindOCR git url: https://github.com/mindspore-lab/mindocr.git



### Prepare Dataset

You can upload your own dataset or associate the project with existing datasets on the platform.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/e5701090-6bbc-4cbb-9c96-07e6ecf1fcde">
</div>
Uploading personal datasets requires setting the available clusters to NPU.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/53098b94-88e3-407b-9062-eac155ee6424">
</div>

### Prepare pretrained model (optional)

To upload pre-trained weights, choose the Model tab of your repository.

<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/be3f1237-05c2-40cb-9587-23f3b5f4d64c">
</div>
During the import of a local model, set the model's framework to MindSpore.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/6bebcca2-72ad-4c09-a5ea-537f84065626">
</div>

### New Training Task

Select Training Task -> New Training Task in the Cloudbrain tab.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/b8081a09-0e99-4c01-b986-860e0cb525f0">
</div>
In computing resources choose Ascend NPU.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/51ef33fa-b03e-4f6c-9dab-786f898816cd">
</div>
Set the training entry point (Start File) and add run parameters.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/eb3aa8cf-81f7-4ab5-a0f8-31aa04fce1e8">
</div>

* To load pre-trained weights, choose the uploaded previously model file in the Select Model field and add `ckpt_dir` to the run parameters. The `ckpt_dir` parameter must have the following path: `/cache/*.ckpt`, where `*` is the model's file name.
* In the AI engine, it is necessary to select MindSpore version 1.9 or higher, and set the start file to `tools/train.py`
* :warning: It is necessary to set `enable_modelarts` to `True` in the run parameters.
* The model's architecture is specified in the `config` file set in the run parameters. The prefix of the file is always `/home/work/user-job-dir/run-version-number`, where `run-version-number` for the newly created training task is usually `V0001`.


### Modify existing training tasks

Click the modify button of an existing training task to modify its parameters and run a new training task.

<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/2efcd351-dd4a-4d6c-b0bd-0f8f82c12996">
</div>

Note: `run-version-number` will change to Parents Version (current run version number) + 1, e.g. `V0002`.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/18af7f35-a4e7-4c78-802d-3aa42b9edc78">
</div>

### View training status

Select a training task to view configuration information, logs, resource occupancy, and download model weights.
<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/983d4525-45c9-4fa4-96f8-9359dc1322d8">
</div>

<div align=center>
<img width="600" src="https://github.com/mindspore-lab/mindocr/assets/52945530/b5e10ca2-b4ef-487f-928e-b5e5a8eeacbc">
</div>


## Reference

[1] Modified from https://github.com/mindspore-lab/mindyolo/blob/master/tutorials/cloud/openi.md
