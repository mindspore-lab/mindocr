import os

PADDLEOCR_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))

# fmt: off
PADDLEOCR_MODELS = {
    "ch_pp_server_det_v2.0": "det/ppocr/ch_det_res18_db_v2.0.yaml",                  # ch_ppocr_server_v2.0_det
    "ch_pp_det_OCRv3": "det/ppocr/ch_PP-OCRv3_det_cml.yaml",                         # ch_PP-OCRv3_det
    "ch_pp_server_rec_v2.0": "rec/ppocr/rec_chinese_common_train_v2.0.yaml",         # ch_ppocr_server_v2.0_rec
    "ch_pp_rec_OCRv3": "rec/ppocr/ch_PP-OCRv3_rec_distillation.yaml",                # ch_PP-OCRv3_rec
    "ch_pp_mobile_cls_v2.0": "cls/ppocr/cls_mv3.yaml",                               # ch_ppocr_mobile_v2.0_cls
    "ch_pp_det_OCRv2": "det/ppocr/ch_PP-OCRv2_det_cml.yaml",                         # ch_PP-OCRv2_det
    "ch_pp_mobile_det_v2.0_slim": "det/ppocr/ch_det_mv3_db_v2.0.yaml",               # ch_ppocr_mobile_slim_v2.0_det
    "ch_pp_mobile_det_v2.0": "det/ppocr/ch_det_mv3_db_v2.0.yaml",                    # ch_ppocr_mobile_v2.0_det
    "en_pp_det_OCRv3": "det/ppocr/ch_PP-OCRv3_det_cml.yaml",                         # en_PP-OCRv3_det
    "ml_pp_det_OCRv3": "det/ppocr/ch_PP-OCRv3_det_cml.yaml",                         # ml_PP-OCRv3_det
    "ch_pp_rec_OCRv2": "rec/ppocr/ch_PP-OCRv2_rec_distillation.yaml",                # ch_PP-OCRv2_rec
    "ch_pp_mobile_rec_v2.0": "rec/ppocr/rec_chinese_lite_train_v2.0.yaml",           # ch_ppocr_mobile_v2.0_rec
    "en_pp_rec_OCRv3": "rec/ppocr/en_PP-OCRv3_rec.yaml",                             # en_PP-OCRv3_rec
    "en_pp_mobile_rec_number_v2.0_slim": "rec/ppocr/rec_en_number_lite_train.yaml",  # en_number_mobile_slim_v2.0_rec
    "en_pp_mobile_rec_number_v2.0": "rec/ppocr/rec_en_number_lite_train.yaml",       # en_number_mobile_v2.0_rec
    "korean_pp_rec_OCRv3": "rec/ppocr/korean_PP-OCRv3_rec.yaml",                     # korean_PP-OCRv3_rec
    "japan_pp_rec_OCRv3": "rec/ppocr/japan_PP-OCRv3_rec.yaml",                       # japan_PP-OCRv3_rec
    "chinese_cht_pp_rec_OCRv3": "rec/ppocr/chinese_cht_PP-OCRv3_rec.yaml",           # chinese_cht_PP-OCRv3_rec
    "te_pp_rec_OCRv3": "rec/ppocr/te_PP-OCRv3_rec.yaml",                             # te_PP-OCRv3_rec
    "ka_pp_rec_OCRv3": "rec/ppocr/ka_PP-OCRv3_rec.yaml",                             # ka_PP-OCRv3_rec
    "ta_pp_rec_OCRv3": "rec/ppocr/ta_PP-OCRv3_rec.yaml",                             # ta_PP-OCRv3_rec
    "latin_pp_rec_OCRv3": "rec/ppocr/latin_PP-OCRv3_rec.yaml",                       # latin_PP-OCRv3_rec
    "arabic_pp_rec_OCRv3": "rec/ppocr/arabic_PP-OCRv3_rec.yaml",                     # arabic_PP-OCRv3_rec
    "cyrillic_pp_rec_OCRv3": "rec/ppocr/cyrillic_PP-OCRv3_rec.yaml",                 # cyrillic_PP-OCRv3_rec
    "devanagari_pp_rec_OCRv3": "rec/ppocr/devanagari_PP-OCRv3_rec.yaml",             # devanagari_PP-OCRv3_rec
    "en_pp_det_psenet_resnet50vd": "det/ppocr/det_r50_vd_pse.yaml",                  # pse_resnet50_vd
    "en_pp_det_dbnet_resnet50vd": "det/ppocr/det_r50_vd_db.yaml",                    # dbnet resnet50_vd
    "en_pp_det_east_resnet50vd": "det/ppocr/det_r50_vd_east.yaml",                   # east resnet50_vd
    "en_pp_det_sast_resnet50vd": "det/ppocr/det_r50_vd_sast_icdar15.yaml",           # sast resnet50_vd
    "en_pp_rec_crnn_resnet34vd": "rec/ppocr/rec_r34_vd_none_bilstm_ctc.yaml",        # crnn resnet34_vd
    "en_pp_rec_rosetta_resnet34vd": "rec/ppocr/rec_r34_vd_none_none_ctc.yaml",       # en_pp_rec_rosetta_resnet34vd
    "en_pp_rec_vitstr_vitstr": "rec/ppocr/rec_vitstr_none_ce.yaml",                  # vitstr
}
# fmt: on
