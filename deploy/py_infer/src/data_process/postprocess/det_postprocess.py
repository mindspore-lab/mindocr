import os
import sys

# add mindocr root path, and import postprocess from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess import det_db_postprocess, det_east_postprocess

__all__ = ["DBPostprocess", "EASTPostprocess"]

DBPostprocess = det_db_postprocess.DBPostprocess
EASTPostprocess = det_east_postprocess.EASTPostprocess
