import os

MINDOCR_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../configs"))

MINDOCR_MODELS = {
    "en_ms_det_dbnet_resnet50": "det/dbnet/db_r50_icdar15.yaml",
    "en_ms_det_dbnetpp_resnet50": "det/dbnet/db++_r50_icdar15.yaml",
    "en_ms_rec_crnn_resnet34": "rec/crnn/crnn_resnet34.yaml"
}
