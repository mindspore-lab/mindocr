# 模型编写指南

本文档提供了编写MindSpore套件中的模型定义文件`model.py`的参考模板，旨在提供一种统一的代码风格。

接下来我们以相对简单的新模型`MLP-Mixer`作为示例。

## 文件头

该文件的**简要描述**。包含模型名称和论文题目。如下所示：

```python
"""
MindSpore implementation of `${MODEL_NAME}`.
Refer to ${PAPER_NAME}.
"""
```

## 模块导入

模块导入分为三种类型。分别为

- Python原生或第三方库。如`import math`、`import numpy as np`等等。应当放在第一梯队。
- MindSpore相关模块。如`import mindspore.nn as nn`、`import mindspore.ops as ops`等等。应当放在第二梯队。
- 套件包内模块。如`from .layers.classifier import ClassifierHead`等等。应当放在第三梯队，并使用相对导入。

示例如下：

```python
import math
from collections import OrderedDict

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init

from .utils import load_pretrained
from .layers.classifier import ClassifierHead
```

仅导入必须的模块或包，避免导入无用包。

## `__all__`

> Python 没有原生的可见性控制，其可见性的维护是靠一套需要大家自觉遵守的”约定“。`__all__` 是针对模块公开接口的一种约定，以提供了”白名单“的形式暴露接口。如果定义了`__all__`，其他文件中使用`from xxx import *`导入该文件时，只会导入` __all__ `列出的成员，可以其他成员都被排除在外。

我们约定模型中对外暴露的接口包括主模型类以及返回不同规格模型的函数， 例如：

```python
__all__ = [
    "MLPMixer",
    "mlp_mixer_s_p32",
    "mlp_mixer_s_p16",
    ...
]
```

其中`"MLPMixer"`是主模型类，`"mlp_mixer_s_p32"`和`"mlp_mixer_s_p16"`等是返回不同规格模型的函数。一般来说子模型，即某`Layer`或某`Block`是不应该被其他文件所共用的。如若此，应当考虑将该子模型提取到`${MINDCLS}/models/layers`下面作为公用模块，如`SEBlock`等。

## 子模型

我们都知道一个深度模型是由多层组成的网络。其中某些层可以组成相同拓扑结构的子模型，我们一般称其为`Layer`或者`Block`，例如`ResidualBlock`等。这种抽象有利于我们理解整个模型结构，也有利于代码的编写。

我们应当通过类注释对子模型进行功能的简要描述。在`MindSpore`中，模型的类继承于`nn.Cell`，一般来说我们需要重载以下两个函数：

- 在`__init__`函数中，我们应当定义模型中需要用到的神经网络层（`__init__ `中的参数要进行参数类型声明，即type hint）。
- 在`construct`函数中我们定义模型前向逻辑。

示例如下：

```python
class MixerBlock(nn.Cell):
    """Mixer Layer with token-mixing MLP and channel-mixing MLP"""

    def __init__(self,
                 n_patches: int,
                 n_channels: int,
                 token_dim: int,
                 channel_dim: int,
                 dropout: float = 0.
                 ) -> None:
        super().__init__()
        self.token_mix = nn.SequentialCell(
            nn.LayerNorm((n_channels,)),
            TransPose((0, 2, 1)),
            FeedForward(n_patches, token_dim, dropout),
            TransPose((0, 2, 1))
        )
        self.channel_mix = nn.SequentialCell(
            nn.LayerNorm((n_channels,)),
            FeedForward(n_channels, channel_dim, dropout),
        )

    def construct(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x
```

在`nn.Cell`类的编写过程中，有两个值得注意的方面

- CellList & SequentialCell

  - CellList is just a container that contains a list of neural network layers(Cell). The Cells contained by it can be properly registered, and will be visible by all Cell methods. We must overwrite the forward calculation, that is, the construct function.


  - SequentialCell is a container than holds a sequential list of layers(Cell). The Cells may have a name(OrderedDict) or not(List). We don't need to implement forward computation, which is done according to the order of the sequential list.

- construct

  - Assert is not supported. [RuntimeError: ParseStatement] Unsupported statement 'Assert'.

  - Usage of single operator。调用算子时（如concat, reshape, mean），使用函数式接口 mindspore.ops.functional (如 output=ops.concat((x1, x2)))，避免先在__init__中实例化原始算子 ops.Primitive  (如self.concat=ops.Concat()) 再在construct中调用（output=self.concat((x1, x2))）。

## 主模型

主模型是论文中所提出的网络模型定义，由多个子模型堆叠而成。它是适用于分类、检测等任务的最顶层网络。它在代码书写上与子模型上基本类似，但有几处不同。

- 类注释。我们应当在此给出论文的题目和链接。另外由于该类对外暴露，我们最好也加上类初始化参数的说明。详见下方代码。
- `forward_features`函数。在函数内对模型的特征网络的运算定义。
- `forward_head`函数。在函数内对模型的分类器的运算进行定义。
- `construct`函数。在函数调用特征网络和分类器的运算。
- `_initialize_weights`函数。我们约定模型参数的随机初始化由该成员函数完成。详见下方代码。

示例如下：

```python
class MLPMixer(nn.Cell):
    r"""MLP-Mixer model class, based on
    `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/abs/2105.01601>`_

    Args:
        depth (int) : number of MixerBlocks.
        patch_size (Union[int, tuple]) : size of a single image patch.
        n_patches (int) : number of patches.
        n_channels (int) : channels(dimension) of a single embedded patch.
        token_dim (int) : hidden dim of token-mixing MLP.
        channel_dim (int) : hidden dim of channel-mixing MLP.
        in_channels(int): number the channels of the input. Default: 3.
        n_classes (int) : number of classification classes. Default: 1000.
    """

    def __init__(self,
                 depth: int,
                 patch_size: Union[int, tuple],
                 n_patches: int,
                 n_channels: int,
                 token_dim: int,
                 channel_dim: int,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.n_channels = n_channels
        # patch with shape of (3, patch_size, patch_size) is embedded to n_channels dim feature.
        self.to_patch_embedding = nn.SequentialCell(
            nn.Conv2d(in_chans, n_channels, patch_size, patch_size, pad_mode="pad", padding=0),
            TransPose(permutation=(0, 2, 1), embedding=True),
        )
        self.mixer_blocks = nn.SequentialCell()
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(n_patches, n_channels, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm((n_channels,))
        self.mlp_head = nn.Dense(n_channels, n_classes)
        self._initialize_weights()

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.to_patch_embedding(x)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        return ops.mean(x, 1)

    def forward_head(self, x: Tensor)-> Tensor:
        return self.mlp_head(x)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)

    def _initialize_weights(self) -> None:
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(init.initializer(init.Normal(0.01, 0), m.weight.shape))
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Constant(0), m.bias.shape))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.gamma.set_data(init.initializer(init.Constant(1), m.gamma.shape))
                if m.beta is not None:
                    m.beta.set_data(init.initializer(init.Constant(0.0001), m.beta.shape))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(init.initializer(init.Normal(0.01, 0), m.weight.shape))
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Constant(0), m.bias.shape))
```

## 规格函数

论文中所提出的模型可能有不同规格的变种，如`channel`的大小、`depth`的大小等等。这些变种的具体配置应该通过规格函数体现，**规格的接口参数： pretrained, num_classes, in_channels 命名要统一**，同时在规格函数内还要进行pretrain loading操作。每一个规格函数对应一种确定配置的规格变种。配置通过入参传入主模型类的定义，并返回实例化的主模型类。另外，还需通过添加装饰器`@register_model`将该模型的此规格注册到包内。

示例如下：

```python
@register_model
def mlp_mixer_s_p16(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    nl, pr, ls, hs, ds, dc = 8, 16, 196, 512, 256, 2048
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds,
     				channel_dim=dc, in_chans=in_chans, n_classes=num_classes, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model

@register_model
def mlp_mixer_b_p32(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    nl, pr, ls, hs, ds, dc = 12, 32, 49, 768, 384, 3072
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds,
                    channel_dim=dc, in_chans=in_chans, n_classes=num_classes, **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
```

## 验证main（可选）

初始编写阶段应当保证模型是可运行的。可通过下述代码块进行基础验证：

```python
if __name__ == '__main__':
    import numpy as np
    import mindspore
    from mindspore import Tensor

    model = mlp_mixer_s_p16()
    print(model)
    dummy_input = Tensor(np.random.rand(8, 3, 224, 224), dtype=mindspore.float32)
    y = model(dummy_input)
    print(y.shape)
```

## 参考示例

- densenet.py
- shufflenetv1.py
- shufflenetv2.py
- mixnet.py
- mlp_mixer.py
