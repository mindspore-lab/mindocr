import os

MMOCR_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))

# fmt: off
MMOCR_MODELS = {
    "en_mm_det_dbnetpp_resnet50": "det/mmocr/dbnetpp_resnet50_fpnc_1200e_icdar2015.yaml",     # dbnet++ resnet50
    "en_mm_det_fcenet_resnet50": "det/mmocr/fcenet_resnet50_fpn_1500e_icdar2015.yaml",        # fcenet resnet50
    "en_mm_rec_nrtr_resnet31": "rec/mmocr/nrtr_resnet31-1by8-1by4_6e_st_mj.yaml",             # nrtr resnet31
    "en_mm_rec_satrn_shallowcnn": "rec/mmocr/satrn_shallow_5e_st_mj.yaml",                    # satrn shallow

}
# fmt: on
