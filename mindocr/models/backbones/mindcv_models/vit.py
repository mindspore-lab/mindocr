# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ViT"""
from typing import List, Optional, Union

import numpy as np

import mindspore as ms
from mindspore import Tensor, nn
from mindspore import ops
from mindspore import ops as P
from mindspore.common.initializer import Normal, initializer
from mindspore.common.parameter import Parameter

from .registry import register_model
from .utils import ConfigDict, load_pretrained

__all__ = [
    "ViT",
    "vit_b_16_224",
    "vit_b_16_384",
    "vit_l_16_224",  # train
    "vit_l_16_384",
    "vit_b_32_224",  # train
    "vit_b_32_384",
    "vit_l_32_224",  # train
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "first_conv": "patch_embed.proj",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "vit_b_16_224": _cfg(url=""),
    "vit_b_16_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_l_16_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_16_224-f02b2487.ckpt"),
    "vit_l_16_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_b_32_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_b_32_224-7553218f.ckpt"),
    "vit_b_32_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_l_32_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_32_224-3a961018.ckpt"),
}


class PatchEmbedding(nn.Cell):
    """
    Path embedding layer for ViT. First rearrange b c (h p) (w p) -> b (h w) (p p c).

    Args:
        image_size (int): Input image size. Default: 224.
        patch_size (int): Patch size of image. Default: 16.
        embed_dim (int): The dimension of embedding. Default: 768.
        input_channels (int): The number of input channel. Default: 3.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = PathEmbedding(224, 16, 768, 3)
    """

    MIN_NUM_PATCHES = 4

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        input_channels: int = 3,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """Path Embedding construct."""
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))

        return x


class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = Attention(768, 12)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        keep_prob: float = 1.0,
        attention_keep_prob: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(head_dim**-0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out


class FeedForward(nn.Cell):
    """
    Feed Forward layer implementation.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features. Default: None.
        out_features (int): The dimension of output features. Default: None
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = FeedForward(768, 3072)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Cell = nn.GELU,
        keep_prob: float = 1.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


class ResidualCell(nn.Cell):
    """
    Cell which implements Residual function:

    $$output = x + f(x)$$

    Args:
        cell (Cell): Cell needed to add residual block.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = ResidualCell(nn.Dense(3,4))
    """

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, keep_prob=None, seed=0):
        super().__init__()
        self.keep_prob = 1 - keep_prob
        seed = min(seed, 0)
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor

        return x


class TransformerEncoder(nn.Cell):
    """
    TransformerEncoder implementation.

    Args:
        dim (int): The dimension of embedding.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        mlp_dim (int): The dimension of MLP hidden layer.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.LayerNorm.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = TransformerEncoder(768, 12, 12, 3072)
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        keep_prob: float = 1.0,
        attention_keep_prob: float = 1.0,
        drop_path_keep_prob: float = 1.0,
        activation: nn.Cell = nn.GELU,
        norm: nn.Cell = nn.LayerNorm,
    ):
        super().__init__()
        drop_path_rate = 1 - drop_path_keep_prob
        dpr = [i.item() for i in np.linspace(0, drop_path_rate, num_layers)]
        attn_seeds = [np.random.randint(1024) for _ in range(num_layers)]
        mlp_seeds = [np.random.randint(1024) for _ in range(num_layers)]

        layers = []
        for i in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  keep_prob=keep_prob,
                                  attention_keep_prob=attention_keep_prob)

            feedforward = FeedForward(in_features=dim,
                                      hidden_features=mlp_dim,
                                      activation=activation,
                                      keep_prob=keep_prob)

            if drop_path_rate > 0:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention,
                                                        DropPath(dpr[i], attn_seeds[i])])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward,
                                                        DropPath(dpr[i], mlp_seeds[i])]))]))
            else:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward]))
                    ])
                )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Transformer construct."""
        return self.layers(x)


class DenseHead(nn.Cell):
    """
    LinearClsHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output. Eg. `ReLU`. Default: None.
        keep_prob (float): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of input.
            Default: 1.0.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        has_bias: bool = True,
        activation: Optional[Union[str, nn.Cell]] = None,
        keep_prob: float = 1.0,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(keep_prob)
        self.classifier = nn.Dense(input_channel, num_classes, has_bias=has_bias, activation=activation)

    def construct(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


class MultilayerDenseHead(nn.Cell):
    """
    MultilayerDenseHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        mid_channel (list): Number of channels in the hidden fc layers.
        keep_prob (list): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of
        input.
        activation (list): activate function applied to the output. Eg. `ReLU`.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        mid_channel: List[int],
        keep_prob: List[float],
        activation: List[Optional[Union[str, nn.Cell]]],
    ) -> None:
        super().__init__()
        mid_channel.append(num_classes)
        assert len(mid_channel) == len(activation) == len(keep_prob), "The length of the list should be the same."

        length = len(activation)
        head = []

        for i in range(length):
            linear = DenseHead(
                input_channel,
                mid_channel[i],
                activation=activation[i],
                keep_prob=keep_prob[i],
            )
            head.append(linear)
            input_channel = mid_channel[i]

        self.classifier = nn.SequentialCell(head)

    def construct(self, x):
        x = self.classifier(x)

        return x


class BaseClassifier(nn.Cell):
    """
    generate classifier to combine the backbone and head
    """

    def __init__(self, backbone, neck=None, head=None):
        super().__init__()
        self.backbone = backbone
        if neck:
            self.neck = neck
            self.with_neck = True
        else:
            self.with_neck = False
        if head:
            self.head = head
            self.with_head = True
        else:
            self.with_head = False

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        if self.with_neck:
            x = self.neck(x)
        if self.with_head:
            x = self.forward_head(x)
        return x


def init(init_type, shape, dtype, name, requires_grad):
    initial = initializer(init_type, shape, dtype).init_data()
    return Parameter(initial, name=name, requires_grad=requires_grad)


class ViT(nn.Cell):
    """
    Vision Transformer architecture implementation.

    Args:
        image_size (int): Input image size. Default: 224.
        input_channels (int): The number of input channel. Default: 3.
        patch_size (int): Patch size of image. Default: 16.
        embed_dim (int): The dimension of embedding. Default: 768.
        num_layers (int): The depth of transformer. Default: 12.
        num_heads (int): The number of attention heads. Default: 12.
        mlp_dim (int): The dimension of MLP hidden layer. Default: 3072.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention layer. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.LayerNorm.
        pool (str): The method of pooling. Default: 'cls'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, 768)`

    Raises:
        ValueError: If `split` is not 'train', 'test' or 'infer'.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = ViT()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 768)

    About ViT:

    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.

    Citation:

    .. code-block::

        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """

    def __init__(
        self,
        image_size: int = 224,
        input_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        keep_prob: float = 1.0,
        attention_keep_prob: float = 1.0,
        drop_path_keep_prob: float = 1.0,
        activation: nn.Cell = nn.GELU,
        norm: Optional[nn.Cell] = nn.LayerNorm,
        pool: str = "cls",
    ) -> None:
        super().__init__()

        # Validator.check_string(pool, ["cls", "mean"], "pool type")

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=input_channels)
        num_patches = self.patch_embedding.num_patches

        if pool == "cls":
            self.cls_token = init(init_type=Normal(sigma=1.0),
                                  shape=(1, 1, embed_dim),
                                  dtype=ms.float32,
                                  name="cls",
                                  requires_grad=True)
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches + 1, embed_dim),
                                      dtype=ms.float32,
                                      name="pos_embedding",
                                      requires_grad=True)
            self.concat = ops.Concat(axis=1)
        else:
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches, embed_dim),
                                      dtype=ms.float32,
                                      name="pos_embedding",
                                      requires_grad=True)
            self.mean = ops.ReduceMean(keep_dims=False)

        self.pool = pool
        self.pos_dropout = nn.Dropout(keep_prob)
        self.norm = norm((embed_dim,))
        self.tile = ops.Tile()
        self.transformer = TransformerEncoder(
            dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            keep_prob=keep_prob,
            attention_keep_prob=attention_keep_prob,
            drop_path_keep_prob=drop_path_keep_prob,
            activation=activation,
            norm=norm,
        )

    def construct(self, x):
        """ViT construct."""
        x = self.patch_embedding(x)

        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (x.shape[0], 1, 1))
            x = self.concat((cls_tokens, x))
            x += self.pos_embedding
        else:
            x += self.pos_embedding
        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = self.mean(x, (1, 2))  # (1,) or (1,2)
        return x


def vit(
    image_size: int,
    input_channels: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    num_classes: int,
    mlp_dim: int,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    activation: nn.Cell = nn.GELU,
    norm: nn.Cell = nn.LayerNorm,
    pool: str = "cls",
    representation_size: Optional[int] = None,
    pretrained: bool = False,
    url_cfg: dict = None,
) -> ViT:
    """Vision Transformer architecture."""
    backbone = ViT(
        image_size=image_size,
        input_channels=input_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        keep_prob=1.0 - dropout,
        attention_keep_prob=1.0 - attention_dropout,
        drop_path_keep_prob=1.0 - drop_path_rate,
        activation=activation,
        norm=norm,
        pool=pool,
    )
    if representation_size:
        head = MultilayerDenseHead(
            input_channel=embed_dim,
            num_classes=num_classes,
            mid_channel=[representation_size],
            activation=["tanh", None],
            keep_prob=[1.0, 1.0],
        )
    else:
        head = DenseHead(input_channel=embed_dim, num_classes=num_classes)

    model = BaseClassifier(backbone=backbone, head=head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load ckpt file.
        load_pretrained(model, url_cfg, num_classes=num_classes, in_channels=input_channels)

    return model


@register_model
def vit_b_16_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 224,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention-dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        num_classes (int): The number of classification. Default: 1000.
        in_channels (int): The number of input channels. Default: 3.
        image_size (int): The input image size. Default: 224 for ImageNet.
        has_logits (bool): Whether has logits or not. Default: False.
        drop_rate (float): The drop out rate. Default: 0.0.s
        drop_path_rate (float): The stochastic depth rate. Default: 0.0.

    Returns:
        ViT network, MindSpore.nn.Cell

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Examples:
        >>> net = vit_b_16_224()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``
    """
    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention-dropout
    config.drop_path_rate = drop_path_rate
    config.pretrained = pretrained
    config.input_channels = in_channels
    config.pool = "cls"
    config.representation_size = 768 if has_logits else None

    config.url_cfg = default_cfgs["vit_b_16_224"]

    return vit(**config)


@register_model
def vit_b_16_384(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 384,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention-dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """construct and return a ViT network"""
    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention-dropout
    config.drop_path_rate = drop_path_rate
    config.pretrained = pretrained
    config.input_channels = in_channels
    config.pool = "cls"
    config.representation_size = 768 if has_logits else None

    config.url_cfg = default_cfgs["vit_b_16_384"]

    return vit(**config)


@register_model
def vit_l_16_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 224,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention-dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """construct and return a ViT network"""

    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention-dropout
    config.drop_path_rate = drop_path_rate
    config.input_channels = in_channels
    config.pool = "cls"
    config.pretrained = pretrained
    config.representation_size = 1024 if has_logits else None

    config.url_cfg = default_cfgs["vit_l_16_224"]

    return vit(**config)


@register_model
def vit_l_16_384(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 384,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention-dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """construct and return a ViT network"""

    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention-dropout
    config.drop_path_rate = drop_path_rate
    config.input_channels = in_channels
    config.pool = "cls"
    config.pretrained = pretrained
    config.representation_size = 1024 if has_logits else None

    config.url_cfg = default_cfgs["vit_l_16_384"]

    return vit(**config)


@register_model
def vit_b_32_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 224,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention-dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """construct and return a ViT network"""
    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention-dropout
    config.drop_path_rate = drop_path_rate
    config.pretrained = pretrained
    config.input_channels = in_channels
    config.pool = "cls"
    config.representation_size = 768 if has_logits else None

    config.url_cfg = default_cfgs["vit_b_32_224"]

    return vit(**config)


@register_model
def vit_b_32_384(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 384,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention_dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """construct and return a ViT network"""
    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention_dropout
    config.drop_path_rate = drop_path_rate
    config.pretrained = pretrained
    config.input_channels = in_channels
    config.pool = "cls"
    config.representation_size = 768 if has_logits else None

    config.url_cfg = default_cfgs["vit_b_32_384"]

    return vit(**config)


@register_model
def vit_l_32_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    image_size: int = 224,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    # attention-dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> ViT:
    """construct and return a ViT network"""
    config = ConfigDict()
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.dropout = drop_rate
    config.attention_dropout = drop_rate  # attention-dropout
    config.drop_path_rate = drop_path_rate
    config.pretrained = pretrained
    config.input_channels = in_channels
    config.pool = "cls"
    config.representation_size = 1024 if has_logits else None

    config.url_cfg = default_cfgs["vit_l_32_224"]

    return vit(**config)
