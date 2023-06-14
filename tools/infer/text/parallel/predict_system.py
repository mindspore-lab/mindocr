import argparse
import os
import sys

from base_predict import BasePredict
from utils_predict import check_args, load_yaml, rescale, save_pipeline_results, update_config
from visualize import VisMode, Visualization

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../../")))

import cv2
import numpy as np

from mindocr.utils.visualize import recover_image


def parse_args():
    """
    If an arg is set in args parser, then use the arg in args parser instead of its counterpart in yaml file.
    """
    parser = argparse.ArgumentParser(description="Evaluation Config", add_help=False)
    parser.add_argument("--raw_data_dir", type=str, help="Directory of raw data to be predicted.")
    parser.add_argument("--det_ckpt_path", type=str, help="Ckpt file path of detection model.")
    parser.add_argument("--rec_ckpt_path", type=str, help="Ckpt file path of recognition model.")
    parser.add_argument(
        "--det_config_path",
        type=str,
        default="configs/det/dbnet/db_r50_icdar15.yaml",
        help="Detection model yaml config file path.",
    )
    parser.add_argument(
        "--rec_config_path",
        type=str,
        default="configs/rec/crnn/crnn_resnet34.yaml",
        help="Recognition model yaml config file path.",
    )
    parser.add_argument(
        "--crop_save_dir",
        type=str,
        default="predict_result/crop",
        required=False,
        help="Saving dir for images cropped during prediction pipeline.",
    )
    parser.add_argument(
        "--result_save_dir",
        type=str,
        default="predict_result/ckpt_pred_result.txt",
        required=False,
        help="Saving dir for pipeline prediction results including bounding boxes and texts.",
    )
    # parser.add_argument('--vis_det_save_dir', type=str, required=False,
    #                     help='Saving dir for visualization of detection results.')
    # parser.add_argument('--vis_result_save_dir', type=str, default='predict_result/vis', required=False,
    #                     help='Saving dir for visualization of pipeline prediction results.')

    args = parser.parse_args()
    args = check_args(args)

    return args


def predict_det(args):
    det_cfg = load_yaml(args.det_config_path)
    det_cfg = update_config(args, det_cfg, "det")

    det_predictor = BasePredict(det_cfg)
    vis_tool = Visualization(VisMode.crop)
    # t0 = time()
    det_pred_outputs = det_predictor()

    ori_img_path_list = det_pred_outputs["img_paths"]
    image_list = det_pred_outputs["pred_images"]  # image_list = [image1, image2, ...]
    box_list = det_pred_outputs["preds"]  # box_list = [[[(boxes1, scores1)]], [[(boxes2, scores2)]], ...]

    # cropped_images_dict = {}
    box_dict = {}
    for idx, image in enumerate(image_list):
        original_img_path = ori_img_path_list[idx].asnumpy()[0]
        # original_img_filename = os.path.splitext(os.path.basename(original_img_path))[0]
        original_img_filename = os.path.basename(original_img_path)

        image = np.squeeze(image.asnumpy(), axis=0)  # TODO: only works when batch size = 1
        image = recover_image(image)
        cropped_images = vis_tool(image, box_list[idx][0][0])
        # cropped_images_dict[original_img_filename] = cropped_images
        box_dict[
            original_img_filename
        ] = {}  # nested dict. {img_0:{img_0_crop_0: box array, img_0_crop_1: box array, ...}, img_1:{...}}

        # save cropped images
        if args.crop_save_dir:
            for i, crop in enumerate(cropped_images):
                crop_save_filename = original_img_filename + "_crop_" + str(i) + ".jpg"
                box_dict[original_img_filename][crop_save_filename] = box_list[idx][0][0][i]
                cv2.imwrite(os.path.join(args.crop_save_dir, crop_save_filename), crop)

    det_pred_outputs["predicted_boxes"] = box_dict
    # t1 = time()
    # det_time = t1 - t0
    # print(f'---det time: {det_time}s')
    # print(f'---det FPS: {len(image_list) / det_time}')
    return det_pred_outputs


def predict_rec(args, det_pred_outputs):
    rec_cfg = load_yaml(args.rec_config_path)
    rec_cfg = update_config(args, rec_cfg, "rec")

    rec_cfg.predict.loader.batch_size = 1  # TODO
    rec_predictor = BasePredict(rec_cfg)
    # t2 = time()

    # rec_predict_image, rec_result, rec_img_path_list = rec_predictor()
    rec_pred_outputs = rec_predictor()
    text_list = [r["texts"][0] for r in rec_pred_outputs["preds"]]

    rec_img_path_list = [os.path.basename(path.asnumpy()[0]) for path in rec_pred_outputs["img_paths"]]
    rec_text_dict = dict(zip(rec_img_path_list, text_list))
    # t3 = time()
    # rec_time = t3 - t2
    # print(f'---rec time: {rec_time}s')
    # print(f"---rec FPS: {len(det_pred_outputs['pred_images']) / rec_time}")

    if args.result_save_dir:
        save_pipeline_results(det_pred_outputs["predicted_boxes"], rec_text_dict, args.result_save_dir)

    # if args.vis_result_save_dir:
    #     vis_tool = Visualization(VisMode.bbox_text)
    #     for idx, image in enumerate(image_list):
    #         # original_img_path = img_path_list[idx].asnumpy()[0]
    #         original_img_filename = os.path.splitext(ori_img_path_list[idx])[0]
    #         pl_vis_filename = original_img_filename + '_pl_vis' + '.jpg'
    #         image = np.squeeze(image.asnumpy(), axis=0)  # TODO: only works when batch size = 1
    #         image = recover_image(image)
    #         box_text = vis_tool(recover_image(image), box_dict[original_img_filename], text_list,
    #                             font_path=args.vis_font_path) # TODO: box_dict
    #         cv2.imwrite(os.path.join(args.vis_result_save_dir, pl_vis_filename), box_text)

    return text_list


def main():
    args = parse_args()
    det_pred_outputs = predict_det(args)
    print("Detection finished!")
    det_pred_outputs = rescale(det_pred_outputs)
    print("Rescale finished!")
    predict_rec(args, det_pred_outputs)
    print("Detection and recognition finished!!!")


if __name__ == "__main__":
    main()
