import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, nn, ops
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer

__all__ = ["LayerNorm", "Linear"]


class LayerNorm(nn.Cell):
    r"""
    A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32, is_self_defined=False):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError(
                "The type of parameter 'param_init_type' should in [float32, float16], "
                "but got the type : {}.".format(type(param_init_type))
            )
        self.is_self_defined = is_self_defined
        if not self.is_self_defined:
            self.layer_norm = ops.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=eps)
        self.gamma = Parameter(
            initializer("ones", normalized_shape, param_init_type), name="gamma", parallel_optimizer=False
        )
        self.beta = Parameter(
            initializer("zeros", normalized_shape, param_init_type), name="beta", parallel_optimizer=False
        )
        self.mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.sub1 = ops.Sub()
        self.sub2 = ops.Sub()
        self.add = ops.Add()
        self.eps = eps
        self.mul = ops.Mul()
        self.add2 = ops.Add()
        self.real_div = ops.RealDiv()

    def construct(self, x):
        # x : batch x seq_length x hidden_size
        if self.is_self_defined:
            mean = self.mean(x, -1)
            diff = self.sub1(x, mean)
            variance = self.mean(self.square(diff), -1)
            variance_eps = self.sqrt(self.add(variance, self.eps))
            output = self.real_div(diff, variance_eps)
            output = self.add2(self.mul(output, self.gamma), self.beta)
        else:
            output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return output


class Linear(nn.Cell):
    r"""
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    3-D tensor.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    where :math:`X` is the input tensors, :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[nn.Cell, str]): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'. Default: None.
        outer_batch (int): The replication number of experts. The replication is effective only when MoE is applied.
            Default: 1.
        expert_group_size (int): The number of tokens in each data parallel group. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: mstype.float16
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, nn.Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    @cell_attr_register
    def __init__(
        self,
        in_channels,
        out_channels,
        weight_init="normal",
        bias_init="zeros",
        has_bias=True,
        activation=None,
        transpose_b=True,
        outer_batch=1,
        expert_group_size=None,
        param_init_type=mstype.float32,
        compute_dtype=mstype.float16,
        skip_redistribution=False,
    ):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not (isinstance(activation, str) or activation is None or issubclass(activation, nn.Cell)):
            raise TypeError(f"For Linear cell, the activation should str type or nn.Cell type, but got {activation}.")
        if isinstance(weight_init, Tensor) and (
            weight_init.ndim != 2 or weight_init.shape[0] != out_channels or weight_init.shape[1] != in_channels
        ):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.outer_batch = outer_batch
        self.expert_group_size = expert_group_size
        self.transpose_b = transpose_b
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = ops.MatMul(transpose_b=transpose_b)
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = ops.Add()
        self.act_name = activation
        if callable(activation):
            self.activation = activation()
        else:
            self.activation = activation
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        if skip_redistribution:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.shape = ops.Shape()

    def construct(self, x):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = ops.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(x)
        x = ops.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output
