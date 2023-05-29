# Guideline for Model Module

## How to Add a New Model in MindOCR

1. Decompose the model into 3 (or 2) modules: backbone, (neck,) head. Neck is usually not involved in recognition tasks.

2. For each module:
   
	a. if it is implemented in MindOCR, skip since you can get the module by the `build_{module}` function . 

	b. if not, please implement it and follow the [module format guideline](#format-guideline-for-writing-a-new-module)

3. Define your model in two ways 

	a. Write a model py file, which includes the model class and specification functions. Please follow the [model format guideline](#format-guideline-for-model-py-file). It is to allows users to invoke a pre-defined model easily, such as `model = build_model('dbnet_resnet50', pretrained=True)`  .

	b. Config the architecture in a yaml file. Please follow the [yaml format guideline](#format-guideline-for-yaml-file) . It is to allows users to modify a base architecture quickly in yaml file. 

4. To verify the correctness of the written model, please run `test_model.py`

``` shell
python tests/ut/test_model.py --config /path/to/yaml_config_file
```
	

## Format Guideline for Writing a New Module

### Backbone
* File naming format: `models/backbones/{task}_{backbone}.py`, e.g, `det_resnet.py`   (since the same backbone for det and rec may differ, the task prefix is necessary)
* Class naming format: **{Task}{BackboneName}{Variant}** e.g. `class DetResNet`
* Class `__init__` args: no limitation, define by model need.
* Class attributes: MUST contain `out_channels` (List), to describe channels of each output features. e.g. `self.out_channels=[256, 512, 1024, 2048]`
* Class `construct` args: x (Tensor)
* Class `construct` return: features (List[Tensor]) for features extracted from different layers in the backbone, feature dim order `[bs, channels, …]`. Expect shape of each feature: `[bs, channels, H, W]`

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
* Class `construct` args: feature (Tensor), extra_input (Optional[Tensor]). The extra_input tensor is only applicable for head that needs recurrent input (e.g., Attention head), or heads with multiple inputs.
* Class `construct` return: prediction (Union(Tensor, Tuple[Tensor])). If there is only one output, return Tensor. If there are multiple outputs, return Tuple of Tensor, e.g., `return output1, output2, output_N`. Note that the order should match the loss function or the postprocess function.


**Note:** if there is no neck in the model architecture like crnn, you can skip writing for neck. `BaseModel` will select the last feature of the features (List(Tensor)) output by Backbone, and forward it Head module.


## Format Guideline for Model Py File

* File naming: `models/{task}_{model_class_name}.py`, e.g., `det_dbnet.py`
* Class naming: {ModelName}, e.g., `class DBNet` 
* Class MUST inherent from `BaseModel`, e.g., `class DBNet(BaseModel)` 
* Spec. function naming: `{model_class_name}_{specifiation}.py`, e.g. `def dbnet_resnet50()` (Note: no need to add task prefix assuming no one model can solve any two tasks)
* Spec. function args: (pretrained=False, **kwargs), e.g. `def dbnet_resnet50(pretrained=False, **kwargs)`. 
* Spec. function return: model (nn.Cell), which is the model instance
* Spec. function decorator: MUST add @register_model decorator, which is to register the model to the supported model list.


After writing and registration, model can be created via the `build_model` func. 
 ``` python
# in a python script
model = build_model('dbnet_resnet50', pretrained=False)
```

## Format Guideline for Yaml File

To define/config the model architecture in yaml file, you should follow the keys in the following examples.


- For models with a neck. 

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

(R is short for Required. D - Depends on model)
