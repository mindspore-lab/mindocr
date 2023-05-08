from . import det_postprocess, east_postprocess, rec_postprocess, cls_postprocess

POSTPROCESS_MAPPING_OPS = {
    # det
    "DBPostprocess": det_postprocess.DBPostprocess,
    "EASTPostprocess": east_postprocess.EASTPostprocess,
    "DistillationDBPostProcess": det_postprocess.DBPostprocess,
    # rec
    "RecCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "CTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "DistillationCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    # cls
    "ClsPostprocess": cls_postprocess.ClsPostprocess,
}
