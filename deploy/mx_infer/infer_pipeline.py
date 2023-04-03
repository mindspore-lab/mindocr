import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../mindocr')))

from deploy.mx_infer.args import get_args
import deploy.mx_infer.pipeline as pipeline


def main():
    args = get_args()
    pipeline.build_pipeline(args)


if __name__ == '__main__':
    main()
