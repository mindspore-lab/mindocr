# MindSpore Model.py Template

This document provides a reference template for writing the model definition file `model.py` in the MindSpore, aiming to provide a unified code style.

Next, let's take `MLP-Mixer` as an example.


## File Header

A brief description of the document. Include model name and paper title. As follows:


```python
"""
MindSpore implementation of `${MODEL_NAME}`.
Refer to ${PAPER_NAME}.
"""
```

## Module Import

There are three types of module imports. Respectively

- Python native or third-party libraries. For example, `import math` and `import numpy as np`. It should be placed in the first echelon.
- MindSpore related modules. For example, `import mindspore.nn as nn` and `import mindspore.ops as ops`. It should be placed in the second echelon.
- The module in the MindCV package. For example, `from .layers.classifier import ClassifierHead`. It should be placed in the third echelon and use relative import.

Examples are as follows:

```python
import math
from collections import OrderedDict

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init

from .utils import load_pretrained
from .layers.classifier import ClassifierHead
```

Only import necessary modules or packages to avoid importing useless packages.

## `__all__`

> Python has no native visibility control, and its visibility is maintained by a set of "conventions" that everyone should consciously abide by `__all__` is a convention for exposing interfaces to modules, and provides a "white list" to expose the interface. If `__all__` is defined, other files use `from xxx import *` to import this file, only the members listed in `__all__` will be imported, and other members can be excluded.

We agree that the exposed interfaces in the model include the main model class and functions that return models of different specifications, such as:

```python
__all__ = [
    "MLPMixer",
    "mlp_mixer_s_p32",
    "mlp_mixer_s_p16",
    ...
]
```

Where `MLPMixer` is the main model class, and `mlp_mixer_s_p32` and `mlp_mixer_s_p16` are functions that return models of different specifications. Generally speaking, a submodel, that is, a `Layer` or a `Block`, should not be shared by other files. If this is the case, you should consider extracting the submodel under `${MINDCLS}/models/layers` as a common module, such as `SEBlock`.

## Submodel

We all know that a depth model is a network composed of multiple layers. Some of these layers can form sub models of the same topology, which we generally call `Layer` or `Block`, such as `ResidualBlock`. This kind of abstraction is conducive to our understanding of the whole model structure, and is also conducive to code writing.

We should briefly describe the function of the sub model through class annotations. In `MindSpore`, the model class inherits from `nn.Cell`. Generally speaking, we need to overload the following two functions:

- In the `__init__` function, we should define the neural network layer that needs to be used in the model (the parameters in `__init__` should be declared with parameter types, that is, type hint).
- In the `construct` function, we define the model forward logic.

Examples are as follows:

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

In the process of compiling the `nn.Cell` class, there are two noteworthy aspects

- CellList & SequentialCell

  - CellList is just a container that contains a list of neural network layers(Cell). The Cells contained by it can be properly registered, and will be visible by all Cell methods. We must overwrite the forward calculation, that is, the construct function.


  - SequentialCell is a container than holds a sequential list of layers(Cell). The Cells may have a name(OrderedDict) or not(List). We don't need to implement forward computation, which is done according to the order of the sequential list.

- construct

  - Assert is not supported. [RuntimeError: ParseStatement] Unsupported statement 'Assert'.

  - Usage of single operator. When calling an operator (such as concat, reshape, mean), use the functional interface mindspore.ops.functional (such as output=ops.concat((x1, x2)) to avoid instantiating the original operator ops.Primary (such as self.Concat()) in __init__ before calling it in construct (output=self.concat((x1, x2)).

## Master Model

The main model is the network model definition proposed in the paper, which is composed of multiple sub models. It is the top-level network suitable for classification, detection and other tasks. It is basically similar to the submodel in code writing, but there are several differences.

- Class annotations. We should give the title and link of the paper here. In addition, since this class is exposed to the outside world, we'd better also add a description of the class initialization parameters. See code below.
- `forward_features` function. The operational definition of the characteristic network of the model in the function.
- `forward_head` function. The operation of the classifier of the model is defined in the function.
- `construct` function. In function call feature network and classifier operation.
- `_initialize_weights` function. We agree that the random initialization of model parameters is completed by this member function. See code below.

Examples are as follows:

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

## Specification Function

The model proposed in the paper may have different specifications, such as the size of the `channel`, the size of the `depth`, and so on. The specific configuration of these variants should be reflected through the specification function. The specification interface parameters: **pretrained, num_classes, in_channels** should be named uniformly. At the same time, the pretrain loading operation should be performed in the specification function. Each specification function corresponds to a specification variant that determines the configuration. The configuration transfers the definition of the main model class through the input parameter, and returns the instantiated main model class. In addition, you need to register this specification of the model in the package by adding the decorator `@register_model`.

Examples are as follows:

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

## Verify Main (Optional)

The initial writing phase should ensure that the model is operational. The following code blocks can be used for basic verification:

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

## Reference Example

- densenet.py
- shufflenetv1.py
- shufflenetv2.py
- mixnet.py
- mlp_mixer.py
