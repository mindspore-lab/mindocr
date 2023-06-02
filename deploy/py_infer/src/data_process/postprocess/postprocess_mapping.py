from . import det_postprocess, rec_postprocess, cls_postprocess

POSTPROCESS_MAPPING_OPS = {
    # det
    "DBPostprocess": det_postprocess.DBPostprocess,
    "EASTPostprocess": det_postprocess.EASTPostprocess,
    "DistillationDBPostProcess": det_postprocess.DBPostprocess,
    "PSEPostprocess": det_postprocess.PSEPostprocess,
    # rec
    "RecCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "CTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "DistillationCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    # cls
    "ClsPostprocess": cls_postprocess.ClsPostprocess,
}
