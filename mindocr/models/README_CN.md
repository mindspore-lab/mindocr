# 模型模块指南

## 如何在MindOCR中添加新模型

1. 将模型分解为3（或2）个模块：主干、(颈部、)头部。颈部通常不参与识别任务。

2. 各模块：
	a. 如果它是在MindOCR中实现的，请跳过，因为您可以通过 `build_{module}` 函数获取模块。

	b. 如果没有，请执行并遵循[模块格式指南](#编写新模块的格式指南)

3. 两种方式定义模型

    a. 编写一个模型py文件，其中包括模型类和规范函数。请遵循[模型格式指南](#py型文件格式指南)。它允许用户方便地调用预定义的模型，例如`model = build_model('dbnet_resnet50', pretrained=True)` 。

    b. 在yaml文件中配置体系结构。请遵循[yaml格式指南](#yaml文件格式指南)。它允许用户在yaml文件中快速修改基本架构。

4. 为验证编写模型的正确性，请在`test_models.py`中添加yaml配置文件路径，修改main函数构建所需模型，然后运行`test_models.py`

``` shell
python tests/ut/test_models.py --config /path/to/yaml_config_file
```

## <span id="编写新模块的格式指南">编写新模块的格式指南</span>

### 主干
* 文件命名格式：`models/backbones/{task}_{backbone}.py`，例如`det_resnet.py`（由于det和rec的同一主干可能不同，因此任务前缀是必需的）
* 类命名格式：**{Task}{BackboneName}{Variant}** 例如 `class DetResNet`
* 类`__init__` args：无限制，按模型需要定义。
* 类属性：必须包含`out_channels` (List)，以描述每个输出功能的通道。例如`self.out_channels=[256, 512, 1024, 2048]`
* 类`construct` args: x (Tensor)
* Class`construct` return：从主干中不同层提取的feature(List[Tensor])，feature dim order`[bs, channels, …]`。每个特征的预期shape：`[bs, channels, H, W]`

### 颈部
* 文件命名格式：`models/necks/{neck_name}.py`，例如，`fpn.py`
* 类命名格式： **{NeckName}**，例如`class FPN`
* 类 `__init__` args:必须包含`in_channels`参数作为第一个位置，例如`__init__(self, in_channels, out_channels=256, **kwargs)`。
* 类属性：必须包含`out_channels`属性，以描述输出特性的通道。例如`self.out_channels=256`
* 类`construct` args：features (List(Tensor))
* 类`construct` return: feature（Tensor）for output feature，feature dim order `[bs, channels, …]`

### 头部
* 文件命名：`models/heads/{head_name}.py`，例如`dbhead.py`
* 类命名：**{HeadName}**，例如 `class DBHead`
* 类`__init__` args:必须包含`in_channels`参数作为第一个位置，例如`__init__(self, in_channels, out_channels=2, **kwargs)`。
* 类`construct` args：feature (Tensor), extra_input (Optional[Tensor])。额外的输入张量仅适用于需要重复输入的头部（例如，注意力头部）或具有多个输入的头部。
* 类`construct` return: prediction (Union(Tensor, Tuple[Tensor]))。如果只有一个输出，则返回张量。如果有多个输出，则返回张量元组，例如 `return output1, output2, output_N`。请注意，顺序应与loss函数或postprocess函数匹配。

**注：** 如果模型体系结构中没有像crnn这样的颈部，可以跳过颈部的编写。`BaseModel`将选择由主干输出的features (List(Tensor))的最后一个特征，并将其转发给Head模块。

## <span id="py型文件格式指南">Py型文件格式指南</span>

* 文件命名：`models/{task}_{model_class_name}.py`，例如`det_dbnet.py`
* 类命名：{ModelName}，例如`class DBNet`
* 类必须是基于`BaseModel`的，例如`class DBNet(BaseModel)`
* 函数命名：`{model_class_name}_{specifiation}.py`，例如`def dbnet_resnet50()`（注意：假设没有一个模型可以解决任何两个任务，则不需要添加任务前缀）
* 函数参数：(pretrained=False, **kwargs)，例如`def dbnet_resnet50(pretrained=False, **kwargs)`。
* 函数返回：model (nn.Cell)，模型实例
* 函数装饰：必须添加 @register_model 装饰，并导入`mindocr/models/__init__.py`中的模型文件，将模型注册到支持的模型列表中。

* 写入和注册后，可以通过`build_model`函数创建模型。
``` python
# in a python script
model = build_model('dbnet_resnet50', pretrained=False)
```

## <span id="yaml文件格式指南">Yaml文件格式指南</span>

要在yaml文件中定义/配置模型体系结构，应该遵循以下示例中的键。

- 对于带有颈部的模型。
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

- 对于无颈部的模型
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
（R-必需。D-取决于模型）
