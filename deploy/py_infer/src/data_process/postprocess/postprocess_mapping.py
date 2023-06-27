from . import cls_postprocess, det_postprocess, rec_postprocess

POSTPROCESS_MAPPING_OPS = {
    # det
    "DBPostprocess": det_postprocess.DBPostprocess,
    "EASTPostprocess": det_postprocess.EASTPostprocess,
    "PSEPostprocess": det_postprocess.PSEPostprocess,
    "SASTPostprocess": det_postprocess.SASTPostprocess,
    "FCEPostprocess": det_postprocess.FCEPostprocess,
    # rec
    "RecCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "RecAttnLabelDecode": rec_postprocess.RecAttnLabelDecode,
    "ViTSTRLabelDecode": rec_postprocess.ViTSTRLabelDecode,
    "AttentionLabelDecode": rec_postprocess.AttentionLabelDecode,
    # cls
    "ClsPostprocess": cls_postprocess.ClsPostprocess,
}
