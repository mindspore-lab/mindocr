from mindspore import _checkparam as validator
from mindspore.ops import ROIAlign
from mindspore.ops.primitive import prim_attr_register


@prim_attr_register
def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num=2, roi_end_mode=1):
    """Initialize ROIAlign"""
    validator.check_value_type("pooled_height", pooled_height, [int], self.name)
    validator.check_value_type("pooled_width", pooled_width, [int], self.name)
    validator.check_value_type("spatial_scale", spatial_scale, [float], self.name)
    validator.check_value_type("sample_num", sample_num, [int], self.name)
    validator.check_value_type("roi_end_mode", roi_end_mode, [int], self.name)
    validator.check_int_range(roi_end_mode, 0, 2, validator.INC_BOTH, "roi_end_mode", self.name)
    self.pooled_height = pooled_height
    self.pooled_width = pooled_width
    self.spatial_scale = spatial_scale
    self.sample_num = sample_num
    self.roi_end_mode = roi_end_mode


def patch_roialign():
    ROIAlign.__init__ = __init__
