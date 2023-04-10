import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)  # src path

from src import infer_args
from src.parallel import ParallelPipeline


def main():
    args = infer_args.get_args()
    parallel_pipeline = ParallelPipeline(args)
    parallel_pipeline.infer_for_images(args.input_images_dir)


if __name__ == '__main__':
    main()
