import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))  # mx_infer path
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))  # tools path

from mx_infer import pipeline_args, pipeline


def main():
    args = pipeline_args.get_args()
    pipeline.build_pipeline(args)


if __name__ == '__main__':
    main()
