"""
Uasage Example:
    $ python tools/infer/text/res_eval.py --draw_img_save_dir inference_results/rec_results.txt \
        --rec_gt_path /Users/Samit/Data/datasets/ic15/rec/test/rec_gt.txt
"""

from config import parse_args
from utils import eval_rec_res

if __name__ == "__main__":
    # parse args
    args = parse_args()

    # pred_fp = 'inference_results/rec_results.txt'
    # pred_fp = 'inference_results/rec_results_serail.txt'
    # rec_gt_fp = '/Users/Samit/Data/datasets/ic15/rec/test/rec_gt.txt'

    pred_fp = args.draw_img_save_dir
    rec_gt_fp = args.rec_gt_path

    perf = eval_rec_res(pred_fp, rec_gt_fp)

    print(perf)
