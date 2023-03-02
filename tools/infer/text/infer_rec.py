"""
Prediction pipeline for text recognition tasks.
"""
from tools.infer.text.args import get_args
from tools.infer.text.infer_pipeline import build_infer_pipeline

if __name__ == '__main__':
    args = get_args()
    build_infer_pipeline(args)
