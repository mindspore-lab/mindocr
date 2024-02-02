# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
"""
Note:
    Transformer Networks. This is interface that is subject to change or deletion.
"""
from __future__ import absolute_import

import inspect
import math
import numpy as np
import os
import sys

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import nn
from mindspore import context
import mindspore
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
sys.path.insert(0, mindocr_path)

from mindocr.models.backbones.mindformers_modules.layers import (
    Linear,
    _args_type_validator_check,
    _check_input_dtype,
    _check_past_none_input_none,
    _valid_type_checks,
    _valid_value_checks,
)

from .op_parallel_config import (
    default_dpmp_config,
    default_moeparallel_config,
    _check_config,
    _Config,
    _PipeLineConfig,
    MoEParallelConfig,
    OpParallelConfig,
)


class EmbeddingOpParallelConfig(_Config):
    r"""
        The parallel config of :class:`VocabEmbedding`
        for the setting data parallel or model parallel for the embedding table.

        Args:
            data_parallel(int): The data parallel way. The input data will be sliced into n parts for embedding layer
                according to this value. Default: 1.
            model_parallel(int): The model parallel way. The embedding table parameters
                will be sliced at 0-th axis according to the model parallel way. Default: 1.
            vocab_emb_dp(bool): Shard embedding in model parallel or data parallel. If True, the embedding lookup
                will be a data parallel style training and model_parallel value will be ignored.  If false, the
                embedding table will be sharded into n parts at the 0-th dimension row slice of the embedding table,
                where the n is the model parallel way determined by this parameter. Default: True

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import EmbeddingOpParallelConfig
            >>> config=EmbeddingOpParallelConfig(data_parallel=1, model_parallel=1, vocab_emb_dp=True)
    """

    def __init__(self, data_parallel=1, model_parallel=1, vocab_emb_dp=True):
        self._dp_mp_config = OpParallelConfig(data_parallel=data_parallel, model_parallel=model_parallel)
        Validator.check_bool(vocab_emb_dp, "vocab_emb_dp")
        self.vocab_emb_dp = vocab_emb_dp

    @property
    def data_parallel(self):
        return self._dp_mp_config.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        self._dp_mp_config.data_parallel = value

    @property
    def model_parallel(self):
        return self._dp_mp_config.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        self._dp_mp_config.model_parallel = value

    @property
    def vocab_emb_dp(self):
        return self._vocab_emb_dp

    @vocab_emb_dp.setter
    def vocab_emb_dp(self, value):
        Validator.check_bool(value, "vocab_emb_dp")
        self._vocab_emb_dp = value

    @property
    def dp_mp_config(self):
        return self._dp_mp_config
    

class TransformerRecomputeConfig(_Config):
    r"""
        TransformerRecomputeConfig for the setting recompute attributes for encoder/decoder layers.

        Args:
            recompute (bool): Enable recomputation of the transformer block or not. Default: False.
            parallel_optimizer_comm_recompute (bool): Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
            mp_comm_recompute (bool): Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            recompute_slice_activation (bool): Slice the cell output which would remains in memory. Default: False.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import TransformerRecomputeConfig
            >>> config=TransformerRecomputeConfig(recompute=True, parallel_optimizer_comm_recompute=True, \
            ...                                   mp_comm_recompute=True, recompute_slice_activation=True)
    """

    def __init__(self, recompute=False, parallel_optimizer_comm_recompute=False,
                 mp_comm_recompute=True, recompute_slice_activation=False):
        Validator.check_bool(recompute, "recompute")
        Validator.check_bool(parallel_optimizer_comm_recompute, "parallel_optimizer_comm_recompute")
        Validator.check_bool(mp_comm_recompute, "mp_comm_recompute")
        Validator.check_bool(recompute_slice_activation, "recompute_slice_activation")
        self._recompute = recompute
        self._parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self._mp_comm_recompute = mp_comm_recompute
        self._recompute_slice_activation = recompute_slice_activation

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        Validator.check_bool(value, "recompute")
        self._recompute = value

    @property
    def parallel_optimizer_comm_recompute(self):
        return self._parallel_optimizer_comm_recompute

    @parallel_optimizer_comm_recompute.setter
    def parallel_optimizer_comm_recompute(self, value):
        Validator.check_bool(value, "parallel_optimizer_comm_recompute")
        self._parallel_optimizer_comm_recompute = value

    @property
    def mp_comm_recompute(self):
        return self._mp_comm_recompute

    @mp_comm_recompute.setter
    def mp_comm_recompute(self, value):
        Validator.check_bool(value, "mp_comm_recompute")
        self._mp_comm_recompute = value

    @property
    def recompute_slice_activation(self):
        return self._recompute_slice_activation

    @recompute_slice_activation.setter
    def recompute_slice_activation(self, value):
        Validator.check_bool(value, "recompute_slice_activation")
        self._recompute_slice_activation = value


default_transformer_recompute_config = TransformerRecomputeConfig()


class TransformerOpParallelConfig(_Config):
    r"""
        TransformerOpParallelConfig for setting parallel configuration, such as the data parallel and model parallel.

        Note:
            Except the recompute argument, other arguments will **not** be effective when the user doesn't set
            auto_parallel_context to `SEMI_AUTO_PARALLEL` or `AUTO_PARALLEL`.
            The micro_batch_num must be greater than or equal to pipeline_stage when training.
            The data_parallel\*model_parallel \*pipeline_stage must be equal or less equal to the device. When setting
            the pipeline stage and optimizer_shard, the config will overwrite the auto_parallel_context. When given the
            8 devices and the data_parallel is 1 and model_parallel is 1, the calculation will be repeated on each
            device.

        Args:
            data_parallel (int): The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel (int): The model parallel way. The parameters of dense layers in MultiheadAttention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            expert_parallel (int): The expert parallel way. This is effective only when MoE (Mixture of Experts)
                is applied. This value specifies the number of partitions to split the experts into.
            pipeline_stage (int): The number of the pipeline stage. Should be a positive value. Default: 1.
            micro_batch_num (int): The micro size of the batches for the pipeline training. Default: 1.
            optimizer_shard (bool): Whether to enable optimizer shard. Default False.
            gradient_aggregation_group (int): The fusion group size of the optimizer state sharding. Default: 4.
            recompute (Union[TransformerRecomputeConfig, bool]): The configuration of recomputation for
                the transformer block. Default: An instance of TransformerRecomputeConfig with default values.
            vocab_emb_dp (bool): Shard embedding in model parallel or data parallel. Default: True.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import TransformerRecomputeConfig
            >>> recompute_config=TransformerRecomputeConfig(recompute=True, parallel_optimizer_comm_recompute=True, \
            ...                                             mp_comm_recompute=True, recompute_slice_activation=True)
            >>> config=TransformerOpParallelConfig(data_parallel=1, model_parallel=1, recompute=recompute_config)
    """

    def __init__(self, data_parallel=1, model_parallel=1, expert_parallel=1, pipeline_stage=1, micro_batch_num=1,
                 recompute=default_transformer_recompute_config,
                 optimizer_shard=False, gradient_aggregation_group=4, vocab_emb_dp=True):
        self.recompute = recompute
        self.optimizer_shard = optimizer_shard
        self.gradient_aggregation_group = gradient_aggregation_group
        self._embed_dp_mp_config = EmbeddingOpParallelConfig(data_parallel=data_parallel, model_parallel=model_parallel,
                                                             vocab_emb_dp=vocab_emb_dp)
        self._pp_config = _PipeLineConfig(pipeline_stage=pipeline_stage, micro_batch_num=micro_batch_num)
        self._moe_config = MoEParallelConfig(data_parallel=data_parallel, model_parallel=model_parallel,
                                             expert_parallel=expert_parallel)

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        if not isinstance(value, TransformerRecomputeConfig) and not isinstance(value, bool):
            raise TypeError(f"recompute must be a TransformerRecomputeConfig/bool, but got {type(value).__name__}.")
        if isinstance(value, bool):
            logger.warning(f"TransformerRecomputeConfig is recommended as the recompute configuration type.")
        self._recompute = value

    @property
    def vocab_emb_dp(self):
        return self._embed_dp_mp_config.vocab_emb_dp

    @vocab_emb_dp.setter
    def vocab_emb_dp(self, value):
        self._embed_dp_mp_config.vocab_emb_dp = value

    @property
    def gradient_aggregation_group(self):
        return self._gradient_aggregation_group

    @gradient_aggregation_group.setter
    def gradient_aggregation_group(self, value):
        Validator.check_positive_int(value, "gradient_aggregation_group")
        self._gradient_aggregation_group = value

    @property
    def micro_batch_num(self):
        return self._pp_config.micro_batch_num

    @micro_batch_num.setter
    def micro_batch_num(self, value):
        self._pp_config.micro_batch_num = value

    @property
    def model_parallel(self):
        return self._embed_dp_mp_config.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        self._embed_dp_mp_config.model_parallel = value
        self._moe_config.model_parallel = value

    @property
    def data_parallel(self):
        return self._embed_dp_mp_config.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        self._embed_dp_mp_config.data_parallel = value
        self._moe_config.data_parallel = value

    @property
    def expert_parallel(self):
        return self._moe_config.expert_parallel

    @expert_parallel.setter
    def expert_parallel(self, value):
        self._moe_config.expert_parallel = value

    @property
    def pipeline_stage(self):
        return self._pp_config.pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        self._pp_config.pipeline_stage = value

    @property
    def optimizer_shard(self):
        return self._optimizer_shard

    @optimizer_shard.setter
    def optimizer_shard(self, value):
        Validator.check_bool(value, "optimizer_shard")
        self._optimizer_shard = value
        context.set_auto_parallel_context(enable_parallel_optimizer=value)

    @property
    def embedding_dp_mp_config(self):
        return self._embed_dp_mp_config

    @property
    def dp_mp_config(self):
        return self._embed_dp_mp_config.dp_mp_config

    @property
    def moe_parallel_config(self):
        return self._moe_config


default_transformer_config = TransformerOpParallelConfig()


class FeedForward(Cell):
    r"""
        The multilayer perceptron with two linear layers with dropout applied at final output. The first linear
        will project the input dimension from hidden_size to ffn_hidden_size. The second linear will project the
        dimension from ffn_hidden_size to hidden_size. The first linear is sharded on the relative dimension,
        and the second linear is sharded on the output dimension. The overview process can be:

        .. math::
            Dropout((xW_1+b_1)W_2 + b_2)

        where the :math:`W_1, W_2, b_1` and :math:`b_2` are trainable parameters.

        Args:
            hidden_size (int): The dimension of the inputs.
            ffn_hidden_size (int): The intermediate hidden size.
            dropout_rate (float): The dropout rate for the second linear's output.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see examples. Default: gelu.
            expert_num (int): The number of experts used in Linear. For the case expert_num > 1, BatchMatMul is used
                and the first dimension in BatchMatMul indicate expert_num. Default: 1.
            expert_group_size (int): The number of tokens in each data parallel group. Default: None. This parameter is
                effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            param_init_type (dtype.Number): The parameter initialization type. Should be mstype.float32 or
                mstype.float16. Default: mstype.float32.
            parallel_config (OpParallelConfig, MoEParallelConfig): The config of parallel setting, see
                `OpParallelConfig` or `MoEParallelConfig`. When MoE is applied, MoEParallelConfig is effective,
                otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            TypeError: `hidden_act` is not a string or nn.Cell.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.
            ValueError: `ffn_hidden_size` is not a multiple of the model parallel way.
            ValueError: `hidden_size` is not a multiple of the model parallel way.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import FeedForward
            >>> from mindspore import dtype as mstype
            >>> from mindspore import Tensor, nn
            >>> import mindspore.ops as ops
            >>> model = FeedForward(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1)
            >>> tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> output = model(tensor)
            >>> print(output.shape)
            (2, 20, 15)
            >>> # Example 2 using custom hidden activation
            >>> class MyActivationNoShard(nn.Cell):
            ...     def __init__(self):
            ...         super(MyActivationNoShard, self).__init__()
            ...         self.add = ops.Add()
            ...     def construct(self, x):
            ...         return self.add(x, 0.1)
            >>> model = FeedForward(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1,
            ...                     hidden_act=MyActivationNoShard)
            >>> tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> output = model(tensor)
            >>> print(output.shape)
            (2, 20, 15)
            >>> # Example 3 using custom hidden activation with activation_shard
            >>> # If user wantss to run on the SEMI/AUTO parallel mode, the custom activation must provide
            >>> # a class function named activation_shard. It accepts the argument parallel_config (OpParallelConfig,
            >>> # MoEParallelConfig) and set the shard for the primitives used in the construct.
            >>> class MyActivationWithShard(nn.Cell):
            ...     def __init__(self):
            ...         super(MyActivationWithShard, self).__init__()
            ...         self.add = ops.Add()
            ...     def construct(self, x):
            ...         return self.add(x, 0.1)
            ...     def activation_shard(self, parallel_config):
            ...         self.add.shard(((parallel_config.data_parallel, parallel_config.model_parallel), ()))
            >>>
            >>> model = FeedForward(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1,
            ...                     hidden_act=MyActivationWithShard)
            >>> tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> output = model(tensor)
            >>> print(output.shape)
            (2, 20, 15)
    """

    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                dropout_rate=Validator.check_non_negative_float,
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "FeedForward"))
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 expert_num=1,
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(FeedForward, self).__init__()
        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            mp = parallel_config.model_parallel
            if expert_num > 1:
                ep = parallel_config.expert_parallel
            else:
                ep = 1
            # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
            dp = parallel_config.data_parallel // ep
            if ffn_hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                                 "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                                 "parallel is {}.".format(ffn_hidden_size, mp))
            if hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                                 "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                                 .format(hidden_size, mp))
            if dropout_rate < 0 or dropout_rate >= 1:
                raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                                 "but got the value : {}.".format(dropout_rate))
            input_size = hidden_size
            output_size = ffn_hidden_size

            # Project to ffn_hidden_size
            self.mapping = Linear(in_channels=input_size,
                                  out_channels=output_size,
                                  activation=hidden_act,
                                  transpose_b=False,
                                  expert_num=expert_num,
                                  expert_group_size=expert_group_size,
                                  outer_batch=dp,
                                  param_init_type=param_init_type)

            # Project back to hidden_size
            self.projection = Linear(in_channels=output_size,
                                     out_channels=input_size,
                                     transpose_b=False,
                                     expert_num=expert_num,
                                     expert_group_size=expert_group_size,
                                     outer_batch=dp,
                                     param_init_type=param_init_type)
            if expert_num > 1:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)))
            self.projection.bias.parallel_optimizer = False
            self.dropout = nn.Dropout(p=dropout_rate)
            self.dropout_3d = nn.Dropout(p=dropout_rate)
            self.dropout_4d = nn.Dropout(p=dropout_rate)
            self.cast = P.Cast()
        else:
            _check_config(parallel_config)
            mp = parallel_config.model_parallel
            if expert_num > 1:
                ep = parallel_config.expert_parallel
            else:
                ep = 1
            # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
            dp = parallel_config.data_parallel // ep
            if ffn_hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                                 "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                                 "parallel is {}.".format(ffn_hidden_size, mp))
            if hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                                 "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                                 .format(hidden_size, mp))
            if dropout_rate < 0 or dropout_rate >= 1:
                raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                                 "but got the value : {}.".format(dropout_rate))
            input_size = hidden_size
            output_size = ffn_hidden_size

            # Project to ffn_hidden_size
            self.mapping = Linear(in_channels=input_size,
                                  out_channels=output_size,
                                  activation=hidden_act,
                                  transpose_b=False,
                                  expert_num=expert_num,
                                  expert_group_size=expert_group_size,
                                  outer_batch=dp,
                                  param_init_type=param_init_type)

            if expert_num > 1:
                self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                                   strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)),
                                   strategy_activation=((dp, ep, 1, mp),))
            else:
                self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                                   strategy_bias=((dp, mp), (mp,)),
                                   strategy_activation=((dp, mp),))
            # Project back to hidden_size
            self.projection = Linear(in_channels=output_size,
                                     out_channels=input_size,
                                     transpose_b=False,
                                     expert_num=expert_num,
                                     expert_group_size=expert_group_size,
                                     outer_batch=dp,
                                     param_init_type=param_init_type)
            if expert_num > 1:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                      strategy_bias=((dp, ep, 1, 1), (1, ep, 1, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                                      strategy_bias=((dp, 1), (1,)))
            self.projection.bias.parallel_optimizer = False
            self.dropout = nn.Dropout(p=dropout_rate)
            self.dropout_3d = nn.Dropout(p=dropout_rate)
            self.dropout_4d = nn.Dropout(p=dropout_rate)
            self.dropout.dropout.shard(((dp, 1),))
            self.dropout_3d.dropout.shard(((dp, 1, 1),))
            self.dropout_4d.dropout.shard(((dp, ep, 1, 1),))
            self.cast = P.Cast()

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, mstype.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)
        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        else:
            output = self.dropout_4d(output)
        return output


class MultiHeadAttention(Cell):
    r"""
        This is an implementation of multihead attention in the paper `Attention is all you need
        <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
        key and value vector with target length, the attention will be performed as the following

        .. math::
               MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

        where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

        if query, key and value tensor is same, then it will be self attention.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            src_seq_length(int): The sequence length of the query vector.
            tgt_seq_length(int): The sequence length of the key and value vector.
            hidden_size(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            compute_dtype(dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            softmax_compute_type(dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            param_init_type(dtype.Number): The parameter initialization type of the module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **query_tensor** (Tensor) - The query vector with shape (batch_size, src_seq_length, hidden_size) or
              (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
              Otherwise, must be (batch_size, 1, hidden_size)
            - **key_tensor** (Tensor) - The key vector with shape (batch_size, tgt_seq_length, hidden_size) or
              (batch_size * tgt_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
              Otherwise, must be (batch_size, 1, hidden_size)
            - **value_tensor** (Tensor) - The value vector with shape (batch_size, tgt_seq_length, hidden_size) or
              (batch_size * tgt_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
              Otherwise, must be (batch_size, 1, hidden_size)
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
              matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
              in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, size_per_head, tgt_seq_length).
              The past calculated key vector. Used for incremental prediction when the use_past is True.
              Default None.
            - **value_past** (Tensor) - Float16 tensor with shape
              (batch_size, num_heads, tgt_seq_length, size_per_head).
              The past calculated value vector. Used for incremental prediction when the use_past is True.
              Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
              shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
              if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, tgt_seq_length),
              (batch_size, num_heads, tgt_seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import MultiHeadAttention
            >>> from mindspore import dtype as mstype
            >>> from mindspore import Tensor
            >>> model = MultiHeadAttention(batch_size=None, hidden_size=15, src_seq_length=20, tgt_seq_length=20,
            ...                            num_heads=3)
            >>> from_tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> to_tensor = Tensor(np.ones((2, 20, 15)), mstype.float16)
            >>> attention_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
            >>> attn_out, past = model(from_tensor, to_tensor, to_tensor, attention_mask)
            >>> print(attn_out.shape)
            (2, 20, 15)
            >>> print(past[0].shape)
            (2, 3, 5, 20)
            >>> print(past[1].shape)
            (2, 3, 20, 5)
            >>> # When use use_past=True, it includes two steps to implement the incremental prediction.
            >>> # Step 1: set is_first_iteration=True, and input the full sequence length's state.
            >>> # We need to prepare the memory parameters for saving key and value states firstly.
            >>> model = MultiHeadAttention(batch_size=2, hidden_size=15, src_seq_length=20, tgt_seq_length=20,
            ...                            num_heads=3, use_past=True)
            >>> key_past = Tensor(np.zeros(shape=(2, 3, 5, 20)), mstype.float16)
            >>> value_past = Tensor(np.zeros(shape=(2, 3, 20, 5)), mstype.float16)
            >>> batch_valid_length = Tensor(np.ones((2,)), mstype.int32)
            >>> # Set is_first_iteration=True to generate the full memory states
            >>> model.add_flags_recursive(is_first_iteration=True)
            >>> attn_out, past = model(from_tensor, to_tensor, to_tensor, attention_mask, key_past, value_past,
            ...                        batch_valid_length)
            >>> print(attn_out.shape)
            (2, 20, 15)
            >>> print(past[0].shape)
            (2, 3, 5, 20)
            >>> print(past[1].shape)
            (2, 3, 20, 5)
            >>> from_tensor = Tensor(np.ones((2, 1, 15)), mstype.float32)
            >>> to_tensor = Tensor(np.ones((2, 1, 15)), mstype.float16)
            >>> attention_mask = Tensor(np.ones((2, 1, 20)), mstype.float16)
            >>> # Step 2: set is_first_iteration=False, and pass the single word to run the prediction rather than the
            >>> # full sequence.
            >>> model.add_flags_recursive(is_first_iteration=False)
            >>> attn_out, past = model(from_tensor, to_tensor, to_tensor, attention_mask, key_past, value_past,
            ...                        batch_valid_length)
            >>> print(attn_out.shape)
            (2, 1, 15)
            >>> print(past[0].shape)
            (2, 3, 5, 20)
            >>> print(past[1].shape)
            (2, 3, 20, 5)
    """

    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "MultiHeadAttention"),
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "MultiHeadAttention"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "MultiHeadAttention"),
                                parallel_config=_valid_type_checks([OpParallelConfig],
                                                                   "MultiHeadAttention"),
                                use_past=Validator.check_bool)
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config):
        super(MultiHeadAttention, self).__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dp = parallel_config.data_parallel
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if batch_size:
            Validator.check_positive_int(batch_size)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the num_heads is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(num_heads, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_channels=hidden_size,
                                     out_channels=hidden_size,
                                     transpose_b=False,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type)
            self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                                  strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                   (parallel_config.model_parallel, 1)))
            self.projection.bias.parallel_optimizer = False
            self.transpose = P.Transpose()
            self.merger_head_transpose = P.Transpose()
            self.reshape = P.Reshape()
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            self.batch_matmul = P.BatchMatMul()
            self.real_div = P.RealDiv()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.add = P.Add()
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.use_past = use_past
            self.dropout = nn.Dropout(p=hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(p=attention_dropout_rate)
            self.softmax = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
            self.expand_dims = P.ExpandDims()

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the num_heads is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(num_heads, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_channels=hidden_size,
                                     out_channels=hidden_size,
                                     transpose_b=False,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type)
            self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                                  strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                   (parallel_config.model_parallel, 1)))
            self.projection.bias.parallel_optimizer = False
            self.transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))
            self.merger_head_transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.reshape = P.Reshape()
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            self.batch_matmul = P.BatchMatMul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.real_div = P.RealDiv().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
            self.sub = P.Sub().shard(
                ((1,), (parallel_config.data_parallel, 1, 1, 1)))
            self.mul = P.Mul().shard(
                ((parallel_config.data_parallel, 1, 1, 1), (1,)))
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.use_past = use_past
            self.dropout = nn.Dropout(p=hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(p=attention_dropout_rate)
            self.dropout.dropout.shard(((parallel_config.data_parallel, 1),))
            self.prob_dropout.dropout.shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax = nn.Softmax().to_float(softmax_compute_type)
            self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
            self.expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))

            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = F.shape(query_tensor)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        ori_dtype = F.dtype(query_tensor)
        query_tensor = F.cast(query_tensor, self.dtype)
        key_tensor = F.cast(key_tensor, self.dtype)
        value_tensor = F.cast(value_tensor, self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = self.transpose(
            F.reshape(
                key, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                      self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and len(F.shape(attention_mask)) == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(-1, 1, 1)), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = self.reducesum(F.cast(self.not_equal(self.slice(key_past, (0, 0, 0, 0),
                                                                               (F.shape(key_tensor)[0], 1, 1,
                                                                                self.src_seq_length),
                                                                               (1, 1, 1, 1)),
                                                                    0), mstype.float32), (1, 2, 3))
                valid_length = F.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = F.reshape(self.attention_mask, (self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        attention = self._attn(query, key, value, attention_mask)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        output = F.cast(output, ori_dtype)
        return output, layer_present

    def _get_batch_size_from_query(self, query):
        r"""Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if len(F.shape(query)) == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return F.shape(query)[0] // self.src_seq_length
        return F.shape(query)[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _check_inputs(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                      value_past=None, batch_valid_length=None):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(query_tensor), "query_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(key_tensor), "key_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(value_tensor), "value_tensor", [mstype.float32, mstype.float16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(F.dtype(attention_mask), "attention_mask", [mstype.float32, mstype.float16],
                               self.cls_name)

        key_is_tensor = isinstance(key_past, Tensor)
        value_is_tensor = isinstance(value_past, Tensor)
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        key_is_default = key_past is None
        value_is_default = value_past is None
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "key_past", self.cls_name, None, key_is_tensor,
                                    key_is_default)
        _check_past_none_input_none(self.use_past, "value_past", self.cls_name, None, value_is_tensor,
                                    value_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)
        if self.use_past:
            _check_input_dtype(F.dtype(key_past), "key_past", [mstype.float16], self.cls_name)
            _check_input_dtype(F.dtype(value_past), "value_past", [mstype.float16], self.cls_name)
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor):
        """convert a nd tensor to a 2d tensor"""
        query_shape = F.shape(query_tensor)
        query_tensor = F.reshape(query_tensor, (-1, query_shape[-1]))
        key_shape = F.shape(key_tensor)
        key_tensor = F.reshape(key_tensor, (-1, key_shape[-1]))
        value_shape = F.shape(value_tensor)
        value_tensor = F.reshape(value_tensor, (-1, value_shape[-1]))

        return query_tensor, key_tensor, value_tensor

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                F.reshape(attention_scores,
                          (shape[0], -1, shape[-1])))
            attention_probs = F.reshape(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = P.Cast()(self.scale_factor, P.DType()(query))
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)

        ori_dtype = P.DType()(score)
        attention_scores = P.Cast()(score, self.softmax_dtype)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = self.reducesum(F.cast(self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                                (F.shape(query)[0], 1, 1,
                                                                                 self.seq_length),
                                                                                (1, 1, 1, 1)),
                                                                     0), mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(F.cast(current_index, mstype.int32), 1)
                index = F.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = F.cast(self.tensor_le(self.range, index), mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub(
                P.Cast()(F.tuple_to_array((1.0,)), P.DType()(attention_scores)),
                P.Cast()(attention_mask, P.DType()(attention_scores)))

            adder = self.mul(multiplu_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = P.Cast()(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


def _get_lambda_func(total_layer=None):
    r"""
    A wrapper function of specifying pipeline stage and gradient aggregation fusion. If the total layer
    is not None, for example, set in the transformer model, the pipeline stage setting function will be
    `(layer_id + 0) // (total_layers / parallel_config.pipeline_stage)` for the encoder and,
    `(layer_id + offset) //
    (total_layers / parallel_config.pipeline_stage)` for the decoder, where `offset` is the layers in the encoder.
    """

    def _set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
        r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs an offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
        """
        # override the layers
        if total_layer:
            layers = total_layer
        # Used for the pipeline's stages setting
        if layers < parallel_config.pipeline_stage:
            raise ValueError(f"layers {layers} must be larger than pipeline stage {parallel_config.pipeline_stage}")

        pp_dis = max(layers // parallel_config.pipeline_stage, 1)
        # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
        pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
        network.pipeline_stage = pp_id

        # Used for optimizer's fusion tag
        dis = max(layers // parallel_config.gradient_aggregation_group, 1)
        network.set_comm_fusion((layer_id + offset) // dis + 1)
        # Used for enabling recomputation of the block
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute:
                network.recompute()
        else:
            if parallel_config.recompute.recompute:
                paralel_op_comm_compute = parallel_config.recompute.parallel_optimizer_comm_recompute
                network.recompute(parallel_optimizer_comm_recompute=paralel_op_comm_compute,
                                  mp_comm_recompute=parallel_config.recompute.mp_comm_recompute,
                                  recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

    return _set_parallel_configure_for_layer
