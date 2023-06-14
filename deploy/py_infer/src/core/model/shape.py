from enum import Enum


class ShapeType(Enum):
    STATIC_SHAPE = 0
    DYNAMIC_SHAPE = 1
    DYNAMIC_BATCHSIZE = 2
    DYNAMIC_IMAGESIZE = 3
