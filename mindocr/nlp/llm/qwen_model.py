from enum import Enum
from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, nn, ops
from mindspore._c_expression import MSContext
from mindspore.common.initializer import initializer
from mindspore.nn.layer.flash_attention import FlashAttention

from mindocr.nlp.llm.base_llm_model import BaseLLMModel
from mindocr.nlp.llm.configs import QwenConfig
from mindocr.nlp.utils.kvcache_mgr import KVCacheMgr, KVCachePreprocess
from mindocr.nlp.utils.layers import Linear
from mindocr.nlp.utils.loss import CrossEntropyLoss


def is_910a():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ["910a", "ascend910"]


class SeqExtendMethod(Enum):
    """Stores the acceptable string identifiers for seq length extend method"""

    PI = "PI"
    NTK = "NTK"
    NONE = "None"


class LlamaEmbedding(nn.Cell):
    """
    Embedding Layer.

    Args:
            - **vocab_size** (int): Size of the dictionary of embeddings.
            - **embedding_size** (int): The size of each embedding vector.
            - **param_init_type** (mstype): The param init type, default mstype.float32.
            - **param_init** (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: "normal".
    Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

    Outputs:
            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
    """

    def __init__(
        self,
        vocab_table_size,
        embedding_size,
        param_init_type=mstype.float32,
        param_init="normal",
        parallel_optimizer=False,
    ):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name="embedding_weight",
            parallel_optimizer=parallel_optimizer,
        )
        self.gather = ops.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output


class FreqsMgr(nn.Cell):
    r"""freqs_cis manager."""

    def __init__(
        self,
        head_dim,
        seq_length=None,
        max_position_embedding=4096,
        rotary_dtype=mstype.float16,
        theta=10000.0,
        scaling_factor=1.0,
        extend_method=SeqExtendMethod.NONE.value,
        is_dynamic=False,
    ):
        super().__init__()
        if seq_length is not None and seq_length > max_position_embedding:
            max_position_embedding = seq_length
        if extend_method == SeqExtendMethod.NTK.value:
            theta *= scaling_factor
        freqs_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (theta ** (freqs_base / head_dim))  # (head_dim // 2, )
        if extend_method == SeqExtendMethod.PI.value:
            t = np.arange(0, max_position_embedding / scaling_factor, 1 / scaling_factor).astype(np.float32)
        else:
            t = np.arange(0, max_position_embedding, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)  # (seq_len, head_dim)
        freqs_sin = np.sin(emb)  # (seq_len, head_dim)
        swap_mask = FreqsMgr.get_swap_mask(head_dim)

        self.head_dim = head_dim
        self.seq_length = max_position_embedding if seq_length is None else seq_length
        self.is_dynamic = is_dynamic
        self.freqs_cos = Tensor(freqs_cos, dtype=rotary_dtype)
        self.freqs_sin = Tensor(freqs_sin, dtype=rotary_dtype)
        self.swap_mask = Tensor(swap_mask, dtype=rotary_dtype)

        self.reshape = ops.Reshape()
        if is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.slice = ops.StridedSlice()
        self.sub = ops.Sub()
        self.gather = ops.Gather()

    def construct(self, seq_length=None):
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        seqlen = seq_length if self.is_dynamic else self.seq_length
        freqs_cos = self.slice(freqs_cos, (0, 0), (seqlen, self.head_dim), (1, 1))
        freqs_sin = self.slice(freqs_sin, (0, 0), (seqlen, self.head_dim), (1, 1))
        return freqs_cos, freqs_sin, self.swap_mask

    def increment(self, batch_valid_length, batch_size):
        freqs_cos = self.reshape(self.gather(self.freqs_cos, batch_valid_length, 0), (batch_size, 1, 1, self.head_dim))
        freqs_sin = self.reshape(self.gather(self.freqs_sin, batch_valid_length, 0), (batch_size, 1, 1, self.head_dim))
        return freqs_cos, freqs_sin, self.swap_mask

    @staticmethod
    def get_swap_mask(head_dim):
        """Swap matrix"""
        zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
        id_block = np.identity(head_dim // 2, dtype=np.float32)
        return np.block([[zero_block, id_block], [-id_block, zero_block]])


class LlamaSiLU(nn.Cell):
    def construct(self, x):
        return ops.silu(x)


class LlamaFeedForward(nn.Cell):
    r"""
    LLaMA FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    def __init__(
        self,
        dim,
        intermediate_size=None,
        hidden_dim=None,
        multiple_of=256,
        hidden_act=LlamaSiLU,
        ffn_dim_multiplier=None,
        compute_dtype=mstype.float16,
        param_init_type=mstype.float32,
        is_dynamic=False,
    ):
        super().__init__()

        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(
                f"For FeedForward cell, the hidden_act should str type or nn.Cell type, but got {hidden_act}."
            )

        if intermediate_size is not None:
            hidden_dim = intermediate_size
        else:
            if ffn_dim_multiplier is not None:
                hidden_dim = int((ffn_dim_multiplier + 0.01) * hidden_dim)
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = ops.Mul()
        self.cast = ops.Cast()
        self.w1 = Linear(
            in_channels=dim,
            out_channels=hidden_dim,
            activation=hidden_act,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

        self.w2 = Linear(
            in_channels=hidden_dim,
            out_channels=dim,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

        self.w3 = Linear(
            in_channels=dim,
            out_channels=hidden_dim,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

    def construct(self, x):
        """Forward process of the FeedForward"""
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x)  # dp,1 -> dp, mp
        hidden = self.w3(x)  # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate)  # dp,mp -> dp, mp
        output = self.w2(hidden)  # dp,mp -> dp, 1
        return output


class LlamaRotaryEmbedding(nn.Cell):
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32, use_rope_slice=False):
        super().__init__(auto_prefix=False)
        self.half_head_dim = head_dim // 2
        self.head_dim = head_dim
        self.dtype = compute_dtype
        self.use_rope_slice = use_rope_slice
        self.is_first_iteration = True

        self.add = ops.Add()
        self.bmm_swap = ops.BatchMatMul()
        self.mul = ops.Mul()
        self.mul_inc = ops.Mul()
        self.neg = ops.Neg()
        self.slice = ops.StridedSlice()
        self.concat = ops.Concat(axis=-1)
        self.shape = ops.Shape()

        self.is_ascend = ms.get_context("device_target") == "Ascend"

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        if self.is_ascend:
            x = self.bmm_swap(x, swap_mask)
        else:
            x = ops.matmul(x, swap_mask)
        return x

    def slice_half(self, x):
        bs, n_head, seq, _ = self.shape(x)
        x1 = self.slice(x, (0, 0, 0, 0), (bs, n_head, seq, self.half_head_dim), (1, 1, 1, 1))
        x2 = self.slice(x, (0, 0, 0, self.half_head_dim), (bs, n_head, seq, self.head_dim), (1, 1, 1, 1))
        x = self.concat((self.neg(x2), x1))
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        original_type = xq.dtype
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        mul = self.mul if self.is_first_iteration else self.mul_inc
        if self.use_rope_slice:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.slice_half(xq), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.slice_half(xk), freqs_sin))
        else:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.rotate_half(xq, swap_mask), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.rotate_half(xk, swap_mask), freqs_sin))

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out


class LLamaAttention(nn.Cell):
    r"""
    This is an implementation of multi head attention in LLaMA.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do incremental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **qkv_has_bias** (bool): Whether Q/K/V in attention has bias or not.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words" state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step"s input tensor, and loop it. Default False.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, head_dim, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                head_dim).
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
                ((batch_size, num_heads, head_dim, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, head_dim)).
    """

    def __init__(
        self,
        batch_size,
        seq_length,
        dim: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        qkv_concat=False,
        compute_dtype=mstype.float16,
        softmax_compute_dtype=mstype.float32,
        rotary_dtype=mstype.float32,
        param_init_type=mstype.float32,
        qkv_has_bias=False,
        use_past=False,
        is_dynamic=False,
        use_kvcache_op=False,
        is_flexible_shape=False,
        use_rope_slice=False,
        use_flash_attention=False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.qkv_concat = qkv_concat

        if self.hidden_size % self.n_head != 0:
            raise ValueError(
                "For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                "of 'n_head', but got the hidden_size is {} and the n_head is {}.".format(self.hidden_size, self.n_head)
            )

        self.inv_norm_factor = Tensor(1.0 / self.head_dim**0.5, dtype=compute_dtype)

        self.shape = ops.Shape()
        self.reshape = ops.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = ops.Transpose()
        self.merger_head_transpose = ops.Transpose()
        self.batch_matmul = ops.BatchMatMul()
        self.batch_matmul_q_k = ops.BatchMatMul(transpose_b=True)
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.softmax = ops.Softmax()
        self.cast = ops.Cast()
        self.cast_attn = ops.Cast()
        self.tile_kv = ops.Tile()
        self.slice_qkv = ops.StridedSlice()

        self.apply_rotary_emb = LlamaRotaryEmbedding(self.head_dim, rotary_dtype, use_rope_slice=use_rope_slice)
        if self.qkv_concat:
            self.w = Linear(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size + self.kv_dim * 2,
                has_bias=qkv_has_bias,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type,
                skip_redistribution=is_dynamic,
            )
        else:
            self.wq = Linear(
                self.hidden_size,
                self.hidden_size,
                has_bias=qkv_has_bias,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type,
                skip_redistribution=is_dynamic,
            )
            self.wk = Linear(
                self.hidden_size,
                self.kv_dim,
                has_bias=qkv_has_bias,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type,
                skip_redistribution=is_dynamic,
            )
            self.wv = Linear(
                self.hidden_size,
                self.kv_dim,
                has_bias=qkv_has_bias,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type,
                skip_redistribution=is_dynamic,
            )
        self.wo = Linear(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(self.head_dim, n_heads, next_block_num=0, high_precision=True)

        if self.use_past:
            self.kvcache_mgr = KVCacheMgr(
                self.n_kv_head,
                self.head_dim,
                max_batch_size=batch_size,
                max_seq_length=seq_length,
                compute_dtype=compute_dtype,
                is_dynamic=is_dynamic,
                use_kvcache_op=use_kvcache_op,
                is_flexible_shape=is_flexible_shape,
            )

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, kvcache_inputs=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)
        # [bs * seq/1, hidden_dim]
        if self.qkv_concat:
            x = self.reshape(x, (-1, x.shape[-1]))
            bs_seq = x.shape[0]
            qkv = self.cast(self.w(x), self.dtype)
            query = self.slice_qkv(qkv, (0, 0), (bs_seq, self.hidden_size), (1, 1))
            key = self.slice_qkv(qkv, (0, self.hidden_size), (bs_seq, self.hidden_size + self.kv_dim), (1, 1))
            value = self.slice_qkv(
                qkv, (0, self.hidden_size + self.kv_dim), (bs_seq, self.hidden_size + self.kv_dim * 2), (1, 1)
            )
        else:
            query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
            key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
            value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp

        if self.use_past and not self.is_first_iteration:
            query = self.reshape(query, (bs, self.n_head, 1, self.head_dim))
            key = self.reshape(key, (bs, self.n_kv_head, 1, self.head_dim))
            value = self.reshape(value, (bs, self.n_kv_head, 1, self.head_dim))
        else:
            query = self.reshape(query, (bs, seq_len, self.n_head, self.head_dim))
            key = self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim))
            value = self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim))
            # [bs, seq/1, n_head/n_kv_head, head_dim]
            query = self.transpose(query, (0, 2, 1, 3))
            key = self.transpose(key, (0, 2, 1, 3))
            value = self.transpose(value, (0, 2, 1, 3))
        # [bs, n_head/n_kv_head, seq/1, head_dim]
        query, key = self.apply_rotary_emb(query, key, freqs_cis)  # dp, mp, 1, 1
        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        if self.use_past:
            key, value = self.kvcache_mgr(key, value, kvcache_inputs)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, mask)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)

        return output

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class LlamaRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (int): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_type: The compute type.
            param_init_type: The layer norm param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32, is_dynamic=False, param_init_type=mstype.float32):
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer("ones", (dim,), dtype=param_init_type), parallel_optimizer=False)

        if ms.get_context("device_target") == "Ascend" and not is_910a() and not is_dynamic:
            self.norm = ops.RmsNorm(eps)
            self.rms_norm = self._rms_norm
        else:
            self.cast = ops.Cast()
            self.mul = ops.Mul()
            self.mul2 = ops.Mul()
            self.square = ops.Square()
            self.mean = ops.ReduceMean(keep_dims=True)
            self.add = ops.Add()
            self.rsqrt = ops.Rsqrt()
            self.rms_norm = self._self_norm

    def _self_norm(self, x):
        original_type = x.dtype
        norm_factor = self.square(self.cast(x, self.compute_type))
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(x, self.cast(norm_factor, original_type))
        output = self.mul2(output, self.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        original_type = x.dtype
        return self.norm(x, self.cast(self.weight, original_type))[0]

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)


class QwenForCausalLM(BaseLLMModel):
    r"""
    Provide qwen training loss or logits through network.
        Args:
            config (QwenConfig): The config of Qwen model.

        Returns:
            Tensor, the loss or logits of the network.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.transformer = QwenModel(config=config)
        self.lm_head = Linear(
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            has_bias=False,
            compute_dtype=config.compute_dtype,
            param_init_type=mstype.float16,
            weight_init="normal",
        )
        self.loss = CrossEntropyLoss()

        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.ignore_token_id = config.ignore_token_id
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.not_equal = ops.NotEqual()
        self.cast = ops.Cast()
        self.add = ops.Add()
        self.reshape = ops.Reshape()
        self.ones = ops.Ones()
        self.slice = ops.StridedSlice()
        self.mul = ops.Mul()
        self.sub_batch_valid_len = ops.Sub()
        self.gather = ops.Gather(1)

    def construct(
        self,
        input_ids,
        labels=None,
        input_position=None,
        position_ids=None,
        attention_mask=None,
        input_embeds=None,
        init_reset=True,
        batch_valid_length=None,
        batch_index=None,
        zactivate_len=None,
    ):
        bsz, seqlen = input_ids.shape
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)

        output = self.transformer(
            tokens,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
        )
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss


class QwenModel(BaseLLMModel):
    """transformer"""

    def __init__(self, config):
        config = QwenConfig(**config)
        super().__init__(config)
        self.dtype = config.compute_dtype
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_layers
        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.seq_length = config.seq_length
        self.pad_token_id = config.pad_token_id
        self.num_attention_heads = config.num_heads
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_kvcache_op = config.use_kvcache_op
        self.is_flexible_shape = config.is_flexible_shape

        self.is_first_iteration = True
        self.use_flash_attention = config.use_flash_attention

        # 1. wte
        self.wte = LlamaEmbedding(
            self.vocab_size, self.embed_dim, param_init_type=config.param_init_type, parallel_optimizer=True
        )

        # 2. drop
        self.drop = nn.Dropout(p=config.emb_dropout_prob)

        # 4. h hidden layers for transformer
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = QwenDecodeLayer(
                config.batch_size,
                config.seq_length,
                layer_id,
                dim=config.hidden_size,
                n_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                norm_eps=config.rms_norm_eps,
                compute_dtype=config.compute_dtype,
                layernorm_compute_dtype=config.layernorm_compute_type,
                softmax_compute_dtype=config.softmax_compute_type,
                rotary_dtype=config.rotary_dtype,
                param_init_type=config.param_init_type,
                ln_param_init_type=config.ln_param_init_type,
                qkv_has_bias=True,
                use_past=config.use_past,
                use_flash_attention=config.use_flash_attention,
            )

            self.layers.append(layer)

        self.freqs_mgr = FreqsMgr(
            head_dim=self.head_dim,
            seq_length=self.seq_length,
            max_position_embedding=config.max_position_embedding,
            rotary_dtype=config.rotary_dtype,
            theta=config.theta,
            scaling_factor=config.scaling_factor,
            extend_method=config.extend_method,
            is_dynamic=config.is_dynamic,
        )
        self.casual_mask = CausalMaskForQwen(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
        )
        self.kvcache_preprocess = KVCachePreprocess(
            max_batch_size=config.batch_size,
            max_seq_length=config.seq_length,
            is_dynamic=config.is_dynamic,
            use_kvcache_op=config.use_kvcache_op,
            is_flexible_shape=config.is_flexible_shape,
        )
        # 5. ln_f
        self.ln_f = LlamaRMSNorm(
            self.embed_dim,
            eps=config.rms_norm_eps,
            compute_type=config.layernorm_compute_type,
            param_init_type=config.ln_param_init_type,
        )

        self.shape = ops.Shape()

    def construct(
        self, input_ids: Tensor, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None
    ):
        """construct"""
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])

        # 1. wte
        hidden_states = self.wte(input_ids)

        # 2. drop
        hidden_states = self.drop(hidden_states)

        # 2. rotary_emb
        bs, seq_len = self.shape(input_ids)
        if not self.use_past:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            mask = self.casual_mask.post_process(mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(
                        self.kvcache_preprocess.range,
                        self.kvcache_preprocess.max_cache_length // bs,
                        batch_valid_length,
                        zactivate_len,
                    )
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)
            mask = self.casual_mask.post_process(mask)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len)

        # 4. hidden_states
        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, kvcache_inputs=kvcache_inputs)

        # 5. ln_f
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class QwenDecodeLayer(nn.Cell):
    def __init__(
        self,
        batch_size,
        seq_length,
        layer_id,
        dim: int = 512,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[int] = None,
        norm_eps: float = 1e-5,
        qkv_concat=False,
        compute_dtype=mstype.float16,
        layernorm_compute_dtype=mstype.float32,
        softmax_compute_dtype=mstype.float32,
        rotary_dtype=mstype.float32,
        param_init_type=mstype.float32,
        ln_param_init_type=mstype.float32,
        use_past=False,
        is_dynamic=False,
        use_kvcache_op=False,
        is_flexible_shape=False,
        use_rope_slice=False,
        use_flash_attention=False,
        qkv_has_bias=True,
    ):
        super().__init__()
        self.batch_size = batch_size

        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past

        self.shape = ops.Shape()
        self.reshape = ops.Reshape().add_prim_attr("skip_redistribution", True)
        self.add = ops.Add()
        self.attention_norm = LlamaRMSNorm(
            self.hidden_size,
            norm_eps,
            compute_type=layernorm_compute_dtype,
            is_dynamic=is_dynamic,
            param_init_type=ln_param_init_type,
        )
        self.ffn_norm = LlamaRMSNorm(
            self.hidden_size,
            norm_eps,
            compute_type=layernorm_compute_dtype,
            is_dynamic=is_dynamic,
            param_init_type=ln_param_init_type,
        )
        self.attention = LLamaAttention(
            batch_size=batch_size,
            seq_length=seq_length,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            qkv_concat=qkv_concat,
            compute_dtype=compute_dtype,
            softmax_compute_dtype=softmax_compute_dtype,
            rotary_dtype=rotary_dtype,
            param_init_type=param_init_type,
            qkv_has_bias=qkv_has_bias,
            use_past=use_past,
            is_dynamic=is_dynamic,
            use_kvcache_op=use_kvcache_op,
            is_flexible_shape=is_flexible_shape,
            use_rope_slice=use_rope_slice,
            use_flash_attention=use_flash_attention,
        )
        self.feed_forward = LlamaFeedForward(
            dim=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_dim=4 * self.hidden_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            is_dynamic=is_dynamic,
        )
        self.feed_forward = QwenFeedForward(
            dim=self.hidden_size,
            intermediate_size=intermediate_size,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )

    def construct(self, x, freqs_cis, mask=None, kvcache_inputs=None):
        """Forward of transformer block."""
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, mask, kvcache_inputs)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out


class QwenFeedForward(nn.Cell):
    r"""
    Qwen FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    def __init__(
        self, dim, intermediate_size=0, compute_dtype=mstype.float16, param_init_type=mstype.float32, is_dynamic=False
    ):
        super().__init__()

        hidden_dim = intermediate_size
        self.dtype = compute_dtype
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = ops.Mul()
        self.cast = ops.Cast()
        self.silu = LlamaSiLU()

        self.w1 = Linear(
            in_channels=dim,
            out_channels=hidden_dim,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

        self.w2 = Linear(
            in_channels=hidden_dim,
            out_channels=dim,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

        self.w3 = Linear(
            in_channels=dim,
            out_channels=hidden_dim,
            has_bias=False,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic,
        )

    def construct(self, x):
        """Forward process of the FeedForward"""
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x)  # dp,1 -> dp, mp
        hidden = self.w3(x)  # dp,1 -> dp, mp
        hidden = self.mul(gate, self.silu(hidden).astype(self.dtype))  # dp,mp -> dp, mp
        output = self.w2(hidden)  # dp,mp -> dp, 1
        return output


class CausalMaskForQwen(nn.Cell):
    r"""Get the Lower triangular matrix from the input_ids.
    [[[1. 0. 0. 0. 0]
      [1. 1. 0. 0. 0]
      [1. 1. 1. 0. 0]
      [1. 1. 1. 1. 0]
      [1. 1. 1. 1. 0]]]"""

    def __init__(
        self, seq_length, compute_type=mstype.float16, is_dynamic=False, pad_token_id=0, use_flash_attention=False
    ):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length))), mstype.float32)

        self.shape = ops.Shape()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.not_equal = ops.NotEqual()
        self.less_equal = ops.LessEqual()
        self.expand_dim = ops.ExpandDims()
        self.slice = ops.StridedSlice()
        self.mul = ops.Mul()
        self.sub = ops.Sub()
        self.mul_post = ops.Mul()
        self.expand_dim_post = ops.ExpandDims()

    def construct(self, tokens):
        """Forward process of the CausalMask"""
        bs = self.shape(tokens)[0]
        seq_len = self.shape(tokens)[1]
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        shape_right = (bs, 1, seq_len)
        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        if not self.is_dynamic:
            lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_triangle = self.expand_dim(lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.mul(mask_right, lower_triangle)
        return attention_mask

    def increment(self, seq_range, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range = self.slice(seq_range, (0, 0, 0), (1, 1, self.shape(zactivate_len)[0]), (1, 1, 1))
        mask = self.less_equal(self.reshape(seq_range, (1, 1, -1)), self.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def increment_slice(self, seq_range, seq_length, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range_mask = self.slice(seq_range, (0, 0, 0), (1, 1, self.shape(zactivate_len)[0]), (1, 1, 1))
        else:
            seq_range_mask = self.slice(seq_range, (0, 0, 0), (1, 1, seq_length), (1, 1, 1))
        mask = self.less_equal(self.reshape(seq_range_mask, (1, 1, -1)), self.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def post_process(self, mask):
        mask = self.sub(self.one, self.cast(mask, self.dtype))
        if not self.use_flash_attention:
            mask = self.expand_dim_post(mask, 1)
            mask = self.mul_post(mask, self.multiply_data)
        return mask
