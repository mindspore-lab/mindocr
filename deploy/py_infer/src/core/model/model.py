import numpy as np

from .backend import MindXModel, LiteModel
from .shape import ShapeType

__all__ = ['Model']

_INFER_BACKEND_MAP = {
    "acl": MindXModel,
    "lite": LiteModel
}


class Model:
    def __init__(self, backend, **kwargs):
        self.model = _INFER_BACKEND_MAP[backend](**kwargs)

    def infer(self, input):
        return self.model.infer(input)

    def get_shape_info(self):
        shape = self.model.input_shape
        gears = self.model.get_gear()
        model_path = self.model.model_path

        if len(shape) != 4:
            raise ValueError(f"Input dim must be 4, but got {len(shape)} for {model_path}.")

        batchsize, channel, height, width = shape

        if channel not in (3,):
            raise ValueError(f"Input channel number must be 3, but got {channel} for {model_path}.")

        # static shape or dynamic shape without gear
        if -1 not in shape:
            return ShapeType.STATIC_SHAPE, (batchsize, channel, height, width)

        if not gears:
            return ShapeType.DYNAMIC_SHAPE, (batchsize, channel, height, width)

        # dynamic shape with gear
        batchsize_list = [gear[0] for gear in gears] if batchsize == -1 else [batchsize]
        height_list = [gear[2] for gear in gears] if height == -1 else [height]
        width_list = [gear[3] for gear in gears] if width == -1 else [width]

        if len(set(batchsize_list)) > 1 and (len(set(height_list)) > 1 or len(set(width_list)) > 1):
            raise ValueError(
                f"Input shape do not support batch_size and image_size as dynamic_dims together for {model_path}.")

        # dynamic_batch_size
        if len(set(batchsize_list)) > 1:
            return ShapeType.DYNAMIC_BATCHSIZE, (tuple(sorted(batchsize_list)), channel, height, width)

        batchsize = batchsize_list[0]

        # dynamic_image_size
        if len(height_list) == 1:
            height_list = height_list * len(width_list)
        if len(width_list) == 1:
            width_list = width_list * len(height_list)

        hw_list = [(h, w) for h, w in zip(height_list, width_list)]
        hw_list.sort(key=lambda x: x[0] * x[1])
        return ShapeType.DYNAMIC_IMAGESIZE, (batchsize, channel, tuple(hw_list))

    def warmup(self):
        shape_type, shape_limit = self.get_shape_info()
        if shape_type == ShapeType.STATIC_SHAPE:
            batchsize, channel, height, width = shape_limit
        elif shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            batchsize_list, channel, height, width = shape_limit
            batchsize = batchsize_list[0]
        elif shape_type == ShapeType.DYNAMIC_IMAGESIZE:
            batchsize, channel, hw_list = shape_limit
            height, width = hw_list[0]
        else:
            # dynamic shape: set width or height = 32 if it is dynamic
            batchsize, channel, height, width = shape_limit
            batchsize = 1 if batchsize == -1 else batchsize
            channel = 3 if channel == -1 else channel
            height = 32 if height == -1 else height
            width = 32 if width == -1 else width

        dummy_tensor = np.random.randn(batchsize, channel, height, width).astype(np.float32)
        self.model.infer([dummy_tensor])

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
