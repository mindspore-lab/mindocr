# Model Writing Guideline

## How to add an ocr model in MindOCR

1. Decompose the model into these modules: backbone, (neck,) head. Neck is usually not involved in recognition tasks.

2. For each module:
   
	a. if the required module exists in MindOCR already, skip to the next moduel since you can invoke `build_{module}()` to get the module. 

	b. if not, following API design to develop this module, referring to the code and the guideline below.

3. Define your model in two ways 
	a. Write a model py file under `mindocr/models/`, which includes the model class and specification functions. Users can invoke your model easily by giving the model name in this way, such as `model = build_model('dbnet_r50', pretrained=True)` 

 	   Please follow the guideline below for model py writing.

	b. Specify the detailed architecture configuration in a yaml file. Users can modify the archtecture easily in this way. Please follow the guideline below for yaml writing.

4. Test it, by running `test_model.py`
	

## Format Guideline for Writng a New Module

### Backbone
* File naming format: `models/backbones/{task}_{backbone}.py`, e.g, `det_resnet.py`   (since the same backbone for det and rec may differ, the task prefix is necessary)
* Class naming format: **{Task}{BackboneName}{Variant}** e.g. `class DetResNet`
* Class `__init__` args: no limitation, define by model need.
* Class attributes: MUST contain `out_channels` (List), to describe channels of each output features. e.g. `self.out_channels=[256, 512, 1024, 2048]`
* Class `construct` args: x (Tensor)
* Class `construct` return: features (List[Tensor]) for features extracted from different layers in the backbone, feature dim order `[bs, channels, …]` 

### Neck

* File naming format: `models/necks/{neck_name}.py`, e.g, `fpn.py` 
* Class naming format: **{NeckName}** e.g. `class FPN`
* Class `__init__` args: MUST contain `in_channels` param as the first position, e.g. `__init__(self, in_channels, out_channels=256, **kwargs)`.  
* Class attributes: MUST contain `out_channels` attribute, to describe channel of the outpu feature. e.g. `self.out_channels=256`
* Class `construct` args: features (List(Tensor))
* Class `construct` return: feature (Tensor) for output feature, feature dim order `[bs, channels, …]` 


### Head

* File naming: `models/heads/{head_name}.py`, e.g., `dbhead.py`
* Class naming: **{HeadName}** e.g. `class DBHead`
* Class `__init__` args: MUST contain `in_channels` param as the first position, e.g. `__init__(self, in_channels, out_channels=2, **kwargs)`.  
* Class `construct` args: feature (Tensor)
* Class `construct` return: no limitation, define by model need, and match the loss func 


Note: it there is no neck in the model architecture like crnn, you can skip it. `BaseModel` will select the last feature in features (List(Tensor)) output by Backbone, and forward it Head.


## Format Guideline for Model Py File

* File naming: `models/{task}_{model_class_name}.py`, e.g., `det_dbnet.py`
* Class naming: {ModelName}, e.g., `class DBNet` 
* Class MUST inherent from `BaseModel`, e.g., `class DBNet(BaseModel)` 
* Spec. function naming: `{model_class_name}_{specifiation}.py`, e.g. `def dbnet_r50()` (Note: no need to add task prefix assuming no one model can solve any two tasks)
* Spec. function args: (pretrained=False, **kwargs), e.g. `def dbnet_r50(pretrained=False, **kwargs)`. 
* Spec. function return: model (nn.Cell), which is the model instance

> Note: Once you define a model specification in model py, you can use this specified architecture in the yaml file for training or inference as follows. 

 ``` python
model:				
  name: dbnet_r50	   	# model specificatio function name	
  pretrained: False

optimizer:
  ...
  
```


## Format Guideline for Arch Config in Yaml File

To define/config the model architecture in yaml file, you should follow the keys in the following examples.


- For models with a neck. 

R is short for Required. D - Depends on model 

``` python
model: 				# R 
  type: det
  backbone: 			# R 
    name: det_resnet50 		# R, backbone specification function name
    pretrained: False
  neck:				# R
    name: FPN			# R, neck class name
    out_channels: 256		# D, neck class __init__ arg 
    #use_asf: True
  head:				# R, head class name
    name: ConvHead 		# D, head class __init__ arg
    out_channels: 2
    k: 50
```

- For models without a neck
``` python
model:				# R
  type: rec
  backbone:			# R
    name: resnet50		# R
    pretrained: False
  head:				# R
    name: ConvHead 		# R
    out_channels: 30		# D
```

