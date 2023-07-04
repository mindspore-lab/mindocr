from typing import List, Tuple

import numpy as np

from .backend import LiteModel, MindXModel
from .shape import ShapeType

__all__ = ["Model"]

_INFER_BACKEND_MAP = {"acl": MindXModel, "lite": LiteModel}


class Model:
    def __init__(self, backend: str, **kwargs):
        self.model = _INFER_BACKEND_MAP[backend](**kwargs)

    def infer(self, input: List[np.ndarray]) -> List[np.ndarray]:
        return self.model.infer(input)

    @property
    def input_num(self) -> int:
        return self.model.input_num

    @property
    def input_dtype(self) -> List:
        return self.model.input_dtype

    @property
    def input_shape(self) -> List[List[int]]:
        return self.model.input_shape

    def get_shape_details(self) -> Tuple[ShapeType, List[Tuple]]:
        if self.model.input_num == 1:
            return self._get_shape_details_for_single_input()
        else:
            return self._get_shape_details_for_multi_input()

    def _get_shape_details_for_multi_input(self):
        all_shape = self.model.input_shape

        # TODO: Only support dynamic/static shape for multi input,
        # shape gear for dynamic batch size/image size is not supported.
        if any(-1 in shape for shape in all_shape):
            return ShapeType.DYNAMIC_SHAPE, all_shape
        else:
            return ShapeType.STATIC_SHAPE, all_shape

    def _get_shape_details_for_single_input(self):
        shape = self.model.input_shape[0]  # input[0].shape
        gears = self.model.get_gear()  # input[0].gear
        model_path = self.model.model_path

        # static shape
        if -1 not in shape:
            return ShapeType.STATIC_SHAPE, [shape]

        # dynamic shape
        if not gears:
            return ShapeType.DYNAMIC_SHAPE, [shape]

        # TODO: Only support NCHW format for single input
        if len(shape) != 4:
            raise ValueError(f"Input dim must be 4, but got {len(shape)} for {model_path}.")

        batchsize, channel, height, width = shape

        # dynamic batch size/image size
        batchsize_list = [gear[0] for gear in gears] if batchsize == -1 else [batchsize]
        height_list = [gear[2] for gear in gears] if height == -1 else [height]
        width_list = [gear[3] for gear in gears] if width == -1 else [width]

        if len(set(batchsize_list)) > 1 and (len(set(height_list)) > 1 or len(set(width_list)) > 1):
            raise ValueError(
                f"Input shape do not support batch_size and image_size as dynamic_dims together for {model_path}."
            )

        # dynamic_batch_size
        if len(set(batchsize_list)) > 1:
            return ShapeType.DYNAMIC_BATCHSIZE, [(tuple(sorted(batchsize_list)), channel, height, width)]

        batchsize = batchsize_list[0]

        # dynamic_image_size
        if len(height_list) == 1:
            height_list = height_list * len(width_list)
        if len(width_list) == 1:
            width_list = width_list * len(height_list)

        hw_list = [(h, w) for h, w in zip(height_list, width_list)]
        hw_list.sort(key=lambda x: x[0] * x[1])
        return ShapeType.DYNAMIC_IMAGESIZE, [(batchsize, channel, tuple(hw_list))]

    def warmup(self):
        scale_divisor = 32
        shape_type, shape_value = self.get_shape_details()

        if shape_type in (ShapeType.STATIC_SHAPE, ShapeType.DYNAMIC_SHAPE):
            warmup_shape = shape_value
            warmup_shape = [tuple(scale_divisor if x == -1 else x for x in shape) for shape in warmup_shape]
        elif shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            batchsize_list, *other_shape = shape_value[0]
            batchsize = batchsize_list[0]
            warmup_shape = [(batchsize, *other_shape)]  # Only single input
        else:  # ShapeType.DYNAMIC_IMAGESIZE
            *other_shape, hw_list = shape_value[0]
            height, width = hw_list[0]
            warmup_shape = [(*other_shape, height, width)]  # Only single input

        dummy_tensor = [np.random.randn(*shape).astype(dtype) for shape, dtype in zip(warmup_shape, self.input_dtype)]
        self.model.infer(dummy_tensor)

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
