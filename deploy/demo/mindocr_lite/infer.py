import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src path

from src import pipeline, pipeline_args


def main():
    args = pipeline_args.get_args()
    pipeline.build_pipeline(args)


if __name__ == "__main__":
    main()
