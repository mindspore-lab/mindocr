"""
Prediction pipeline for text detection tasks.
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from tools.infer.text.args import get_args
from deploy.infer_pipeline.pipeline import build_pipeline

if __name__ == '__main__':
    args = get_args()
    build_pipeline(args)
