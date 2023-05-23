from ..preprocess import transforms

PREPROCESS_SKIP_OPS = ["DetLabelEncode", "RecCTCLabelEncode", "CTCLabelEncode"]

PREPROCESS_MAPPING_OPS = {
    "DecodeImage": transforms.DecodeImage,
    "NormalizeImage": transforms.NormalizeImage,
    "ToCHWImage": transforms.ToCHWImage,
    "GridResize": transforms.ResizeImage,
    "ResizeImage": transforms.ResizeImage,
    "ScalePadImage": transforms.ScalePadImage,
    "RecResizeImg": transforms.RecResizeImg,
    "ClsResizeImg": transforms.ClsResizeImg
}
