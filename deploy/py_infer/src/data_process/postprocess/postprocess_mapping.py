from . import cls_postprocess, det_postprocess, rec_postprocess

POSTPROCESS_MAPPING_OPS = {
    # det
    "DBPostprocess": det_postprocess.DBPostprocess,
    "EASTPostprocess": det_postprocess.EASTPostprocess,
    "DistillationDBPostProcess": det_postprocess.DBPostprocess,
    "PSEPostprocess": det_postprocess.PSEPostprocess,
    "SASTPostprocess": det_postprocess.SASTPostprocess,
    "FCEPostprocess": det_postprocess.FCEPostprocess,
    # rec
    "RecCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "CTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "DistillationCTCLabelDecode": rec_postprocess.RecCTCLabelDecode,
    "ViTSTRLabelDecode": rec_postprocess.ViTSTRLabelDecode,
    # cls
    "ClsPostprocess": cls_postprocess.ClsPostprocess,
}
