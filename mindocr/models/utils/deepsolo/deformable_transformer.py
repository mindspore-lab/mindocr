import copy
import math

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops

from .deepsolo_config import DEBUG

if DEBUG:
    import sys
    import os
    mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    sys.path.insert(0, mindocr_path)
    from tests.deepsolo.test_nn_mock import Dense_mock as Dense
else:
    from mindspore.nn import Dense

def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, _, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    split_points = [H_ * W_ for H_, W_ in value_spatial_shapes]
    split_points = split_points[:-1]
    split_points_size = len(split_points)
    for i in range(1, split_points_size):
        split_points[i] = split_points[i - 1] + split_points[i]
    split_points = [int(one.asnumpy()) for one in split_points]
    value_list = mnp.split(value, split_points, axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        n1, n2, n3, n4 = value_list[lid_].shape
        value_l_ = value_list[lid_].reshape(n1, n2, n3 * n4)
        value_l_ = ops.transpose(value_l_, (0, 2, 1))
        value_l_ = value_l_.reshape(N_ * M_, D_, int(H_.asnumpy()), int(W_.asnumpy()))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]
        sampling_grid_l_ = ops.transpose(sampling_grid_l_, (0, 2, 1, 3, 4))
        n1, n2, n3, n4, n5 = sampling_grid_l_.shape
        sampling_grid_l_ = sampling_grid_l_.reshape(n1 * n2, n3, n4, n5)
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = ops.transpose(attention_weights, (0, 2, 1, 3, 4)).reshape(N_ * M_, 1, Lq_, L_ * P_)
    sampling_value_list = mnp.stack(sampling_value_list, axis=-2)
    n1, n2, n3, n4, n5 = sampling_value_list.shape
    sampling_value_list = sampling_value_list.reshape(n1, n2, n3, n5 * n4)
    output = (sampling_value_list * attention_weights).sum(-1).view((N_, M_ * D_, Lq_))
    output = ops.transpose(output, (0, 2, 1))
    return output


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Cell):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            print(
                "[Warning], You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )
        self.im2col_step = 64
        ###
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = Dense(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = Dense(d_model, n_heads * n_levels * n_points)
        self.value_proj = Dense(d_model, d_model)
        self.output_proj = Dense(d_model, d_model)

    def construct(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        tmp_len = copy.deepcopy(int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()))

        assert tmp_len == Len_in
        ####
        value = self.value_proj(input_flatten)  # ZHQ DEBUG: -34473288
        if input_padding_mask is not None:
            # value = value.masked_fill(input_padding_mask[..., None], float(0))
            value = ops.masked_fill(value, input_padding_mask[..., None], 0.0)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)  # ZHQ DEBUG: -2.15522e+07
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)  # ZHQ DEBUG: 1.08526e+09

        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)   # ZHQ DEBUG: 5.42034e+08

        softmax = ops.Softmax()
        attention_weights = softmax(attention_weights).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = ops.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        output = ms_deform_attn_core(value, input_spatial_shapes, sampling_locations, attention_weights)   # ZHQ DEBUG: -6.5228e+06

        output = self.output_proj(output)   # ZHQ DEBUG: -8.1424e+08

        return output