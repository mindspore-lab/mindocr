import math

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from ...utils.box_utils import tensor
from .patch import patch_roialign

patch_roialign()

pooler_type_map = {
    "ROIAlign": 0,
    "ROIAlignV2": 2
}


class RoIExtractor(nn.Cell):
    """
    Extract RoI features from multiple feature map.

    Args:
        resolution (int) - RoI resolution.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 resolution,
                 featmap_strides,
                 pooler_sampling_ratio=0,
                 pooler_type="ROIAlign",
                 finest_scale=224,
                 canonical_level=4):
        super(RoIExtractor, self).__init__()
        self.finest_scale = finest_scale
        self.canonical_level = canonical_level
        self.roi_layers = []
        self.num_levels = len(featmap_strides)
        self.min_level = int(-(math.log2(1 / featmap_strides[0])))
        self.max_level = int(-(math.log2(1 / featmap_strides[-1])))
        self.resolution = resolution
        for s in featmap_strides:
            self.roi_layers.append(
                ops.ROIAlign(pooled_height=resolution,
                             pooled_width=resolution,
                             spatial_scale=1 / s,
                             sample_num=pooler_sampling_ratio,
                             roi_end_mode=pooler_type_map[pooler_type])
            )
        self.featmap_strides = featmap_strides
        self.temp_roi = ms.Tensor(
            np.array([0, 0, featmap_strides[-1] + 1, featmap_strides[-1] + 1]).astype(np.float32).reshape(1, 4)
        )

    def log2(self, value):
        return ops.log(value + 1e-4) / ops.log(tensor(2, value.dtype))

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = ops.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = ops.floor(self.canonical_level + ops.log2(scale / self.finest_scale + 1e-8))
        target_lvls = target_lvls.clamp(min=self.min_level, max=self.max_level) - self.min_level
        return target_lvls

    def construct(self, features, rois, rois_mask):
        # rois shape is [batch_size, self.post_nms_top_n, 4] 4 is (x0, y0, x1, y1)
        batch_size, num_sample, _ = rois.shape
        batch_c = ops.repeat_elements(ops.arange(batch_size).astype(rois.dtype), num_sample, axis=0).reshape(-1, 1)
        rois = rois.reshape(batch_size * num_sample, 4)
        rois_mask = rois_mask.reshape(batch_size * num_sample, 1).astype(ms.bool_)
        rois = ops.select(
            ops.tile(rois_mask, (1, 4)), rois, ops.tile(self.temp_roi.astype(rois.dtype), (batch_size * num_sample, 1))
        )
        rois = ops.concat((batch_c, rois), 1)
        out_channel = features[0].shape[1]
        target_lvls = self.map_roi_levels(rois, self.num_levels).reshape(batch_size * num_sample, 1)
        res = ops.zeros((batch_size * num_sample, out_channel, self.resolution, self.resolution), features[0].dtype)
        for i in range(self.num_levels):
            mask = ops.logical_and(target_lvls == i, rois_mask)
            mask = ops.tile(mask.reshape((-1, 1, 1, 1)), (1, out_channel, self.resolution, self.resolution))
            roi_feats_t = self.roi_layers[i](features[i], rois)
            res = ops.select(mask, roi_feats_t, res)
        return res.reshape(batch_size, num_sample, out_channel, self.resolution, self.resolution)
