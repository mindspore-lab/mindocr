# The code is based on
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/anchor_generator.py

import math
import mindspore as ms
from mindspore import ops, nn


class AnchorGenerator(nn.Cell):
    """
    Generate anchors for RCNN

    Args:
        anchor_sizes (list[float]): The anchor sizes at each feature point.
        aspect_ratios (list[float]): The aspect ratios at each feature point.
        strides (list[float]): The strides of feature maps of anchors.
        offset (float): The offset of anchors.
    """

    def __init__(
        self,
        anchor_sizes=[[64], [128], [256], [512]],
        aspect_ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64],
        variance=[1.0, 1.0, 1.0, 1.0],
        offset=0.0,
    ):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.variance = variance
        self.cell_anchors = self.calculate_anchors(len(strides))
        self.offset = offset

    def broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):
            return [params] * num_features
        if len(params) == 1:
            return list(params) * num_features
        return params

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return ms.Tensor(anchors, ms.float32)

    def calculate_anchors(self, num_features):
        sizes = self.broadcast_params(self.anchor_sizes, num_features)
        aspect_ratios = self.broadcast_params(self.aspect_ratios, num_features)
        cell_anchors = [self.generate_cell_anchors(s, a) for s, a in zip(sizes, aspect_ratios)]
        return cell_anchors

    def create_grid_offsets(self, size, stride, offset):
        grid_height, grid_width = size[0], size[1]
        shifts_x = ms.numpy.arange(offset * stride, grid_width * stride, step=stride, dtype=ms.float32)
        shifts_y = ms.numpy.arange(offset * stride, grid_height * stride, step=stride, dtype=ms.float32)
        # shift_x, shift_y = ops.meshgrid((shifts_x, shifts_y))
        shift_x, shift_y = ops.meshgrid(shifts_x, shifts_y)
        shift_x = ops.reshape(shift_x, (-1,))
        shift_y = ops.reshape(shift_y, (-1,))
        return shift_x, shift_y

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = self.create_grid_offsets(size, stride, self.offset)
            shifts = ops.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
            shifts = ops.reshape(shifts, (-1, 1, 4))
            base_anchors = ops.reshape(base_anchors, (1, -1, 4))
            anchor = ops.reshape(shifts + base_anchors, (-1, 4))
            anchors.append(anchor)

        return anchors

    def construct(self, grid_sizes):
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps

    @property
    def num_anchors(self):
        """
        Returns:
            int: number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                For FPN models, `num_anchors` on every feature map is the same.
        """
        return len(self.cell_anchors[0])


if __name__ == "__main__":
    anchors = AnchorGenerator()(((192, 320), (96, 160), (48, 80), (24, 40)))
    for a in anchors:
        print(a.shape, a[100:120])