import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../")))

from pipeline import infer_args  # noqa
from pipeline.parallel_pipeline import ParallelPipeline  # noqa


def main():
    args = infer_args.get_args()
    parallel_pipeline = ParallelPipeline(args)
    parallel_pipeline.start_pipeline()
    parallel_pipeline.infer_for_images(args.input_images_dir, task_id=0)
    parallel_pipeline.stop_pipeline()


if __name__ == "__main__":
    main()
