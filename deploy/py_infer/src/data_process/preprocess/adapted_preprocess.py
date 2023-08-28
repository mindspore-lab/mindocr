# The data preprocess customized for inference needs to be imported here

from .transforms import *  # noqa

# TODO: remove gear_supported_list
# some operations that support dynamic image size,
gear_supported_list = [
    "DetResize",
    "DetResizeNormForInfer",
    "SVTRRecResizeImg",
    "RecResizeNormForInfer",
    "RecResizeNormForViTSTR",
    "RecResizeNormForMMOCR",
]
