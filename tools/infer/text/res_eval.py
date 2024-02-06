"""
Uasage Example:
    $ python tools/infer/text/res_eval.py --draw_img_save_dir inference_results/rec_results.txt \
        --rec_gt_path /Users/Samit/Data/datasets/ic15/rec/test/rec_gt.txt
"""

import argparse

from utils import eval_rec_res


def parse_args():
    parser = argparse.ArgumentParser(description="Result Evaluation Config Args")

    parser.add_argument(
        "--draw_img_save_dir",
        type=str,
        default="./inference_results",
        help="Dir to save visualization and detection/recogintion/system prediction results",
    )
    parser.add_argument(
        "--rec_gt_path", type=str, default=None, help="Path to ground truth labels of the recognition result"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse args
    args = parse_args()
    pred_fp = args.draw_img_save_dir
    rec_gt_fp = args.rec_gt_path

    perf = eval_rec_res(pred_fp, rec_gt_fp)

    print(perf)
