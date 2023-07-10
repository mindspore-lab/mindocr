from mindspore.dataset.vision import HWC2CHW, Normalize

from . import transforms


class MSWrapper:
    def __init__(self, transform, **params):
        self._transform = transform(**params)

    def __call__(self, data: dict) -> dict:
        data["image"] = self._transform(data["image"])
        return data


# other ops node will be skipped
PREPROCESS_MAPPING_OPS = {
    # general
    "DecodeImage": transforms.DecodeImage,
    "Normalize": lambda **x: MSWrapper(Normalize, **x),
    "HWC2CHW": lambda: MSWrapper(HWC2CHW),
    # det
    "DetResize": transforms.DetResize,
    "DetResizeNormForInfer": transforms.DetResizeNormForInfer,
    # rec
    "SVTRRecResizeImg": transforms.SVTRRecResizeImg,
    "RecResizeNormForInfer": transforms.RecResizeNormForInfer,
    "RecResizeNormForViTSTR": transforms.RecResizeNormForViTSTR,
    "RecResizeNormForMMOCR": transforms.RecResizeNormForMMOCR,
    # cls
    "ClsResizeNormForInfer": transforms.ClsResizeNormForInfer,
}
