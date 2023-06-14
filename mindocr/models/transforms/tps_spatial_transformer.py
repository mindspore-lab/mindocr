import itertools
from typing import Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


def grid_sample(input: Tensor, grid: Tensor, canvas: Optional[Tensor] = None) -> Tensor:
    output = ops.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = ops.ones_like(input)
        output_mask = ops.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


def build_output_control_points(
    num_control_points: int, margins: Tuple[float, float]
) -> np.ndarray:
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
    return output_ctrl_pts


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(
    input_points: np.ndarray, control_points: np.ndarray
) -> np.ndarray:
    N = input_points.shape[0]
    M = control_points.shape[0]
    pairwise_diff = np.reshape(input_points, (N, 1, 2)) - np.reshape(
        control_points, (1, M, 2)
    )
    pairwise_dist = np.sum(pairwise_diff**2, axis=2)
    repr_matrix = 0.5 * pairwise_dist * np.log(pairwise_dist, where=pairwise_dist != 0)
    return repr_matrix


class TPSSpatialTransformer(nn.Cell):
    def __init__(
        self,
        output_image_size: Tuple[int, int] = [32, 100],
        num_control_points: int = 20,
        margins: Tuple[float, float] = [0.05, 0.05],
    ):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points, margins)
        N = num_control_points

        # create padded kernel matrix
        forward_kernel = np.zeros((N + 3, N + 3))
        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points
        )
        forward_kernel[:N, :N] = target_control_partial_repr
        forward_kernel[:N, -3] = 1
        forward_kernel[-3, :N] = 1
        forward_kernel[:N, -2:] = target_control_points
        forward_kernel[-2:, :N] = np.transpose(target_control_points, axes=(1, 0))

        # compute inverse matrix
        inverse_kernel = np.linalg.inv(forward_kernel)

        # create target cordinate matrix
        HW = self.target_height * self.target_width
        target_coordinate = list(
            itertools.product(range(self.target_height), range(self.target_width))
        )
        target_coordinate = np.array(target_coordinate)

        Y, X = np.split(
            target_coordinate, indices_or_sections=target_coordinate.shape[1], axis=1
        )
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = np.concatenate(
            [X, Y], axis=1
        )  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(
            target_coordinate, target_control_points
        )
        target_coordinate_repr = np.concatenate(
            [
                target_coordinate_partial_repr,
                np.ones((HW, 1)),
                target_coordinate,
            ],
            axis=1,
        )

        # register precomputed matrices
        self.inverse_kernel = Tensor(inverse_kernel, dtype=ms.float32)
        self.padding_matrix = ops.zeros((1, 3, 2), ms.float32)
        self.target_coordinate_repr = Tensor(target_coordinate_repr, dtype=ms.float32)
        self.target_control_points = Tensor(target_control_points, dtype=ms.float32)

    def construct(
        self, input: Tensor, source_control_points: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size = ops.shape(source_control_points)[0]

        padding_matrix = ops.tile(self.padding_matrix, (batch_size, 1, 1))
        Y = ops.concat([source_control_points, padding_matrix], axis=1)
        mapping_matrix = ops.matmul(self.inverse_kernel, Y)
        source_coordinate = ops.matmul(self.target_coordinate_repr, mapping_matrix)
        grid = ops.reshape(
            source_coordinate,
            (-1, self.target_height, self.target_width, 2),
        )
        grid = grid.clip(0, 1)  # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate
