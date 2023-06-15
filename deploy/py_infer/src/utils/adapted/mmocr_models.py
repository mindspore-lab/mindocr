import os

MMOCR_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))

# fmt: off
MMOCR_MODELS = {
    "en_mm_det_dbnetpp_resnet50": "det/mmocr/dbnetpp_resnet50_fpnc_1200e_icdar2015.yaml",     # dbnet++ resnet50
    "en_mm_det_fcenet_resnet50": "det/mmocr/fcenet_resnet50_fpn_1500e_icdar2015.yaml",        # fcenet resnet50
}
# fmt: on
