import os

PADDLEOCR_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))


PADDLEOCR_MODELS = {
    "ch_pp_server_det_v2.0": "det/ppocr/ch_det_res18_db_v2.0.yaml",            # ch_ppocr_server_v2.0_det
    "ch_pp_det_v3": "det/ppocr/ch_PP-OCRv3_det_cml.yaml",                      # ch_PP-OCRv3_det
    "ch_pp_server_rec_v2.0": "rec/ppocr/rec_chinese_common_train_v2.0.yaml",   # ch_ppocr_server_v2.0_rec
    "ch_pp_rec_v3": "rec/ppocr/ch_PP-OCRv3_rec_distillation.yaml",             # ch_PP-OCRv3_rec
    "ch_pp_mobile_cls_v2.0": "cls/ppocr/cls_mv3.yaml"                          # ch_ppocr_mobile_v2.0_cls
}
