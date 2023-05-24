from ..postprocess import db_postprocess, rec_postprocess, cls_postprocess

POSTPROCESS_MAPPING_OPS = {
    "DBPostprocess": db_postprocess.DBPostprocess,
    "RecCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "ClsPostprocess": cls_postprocess.ClsPostprocess,
    "DistillationDBPostProcess": db_postprocess.DBPostprocess,
    "DistillationCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "CTCLabelDecode": rec_postprocess.RecCTCLabelDecode
}
