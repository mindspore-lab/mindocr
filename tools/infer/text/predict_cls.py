"""
Text classification inference

Example:
    $ python tools/infer/text/predict_cls.py  --image_dir {img_dir_or_img_path} --cls_algorithm MV3
"""
import os
import sys
from time import time

import numpy as np
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from tqdm import tqdm
from utils import get_ckpt_file, get_image_paths

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr import build_model
from mindocr.utils.logger import Logger
from mindocr.utils.visualize import show_imgs

# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {"MV3": "cls_mobilenet_v3_small_100_model"}
_logger = Logger("mindocr")


class DirectionClassifier(object):
    def __init__(self, args):
        self.batch_num = args.cls_batch_num
        self.batch_mode = args.cls_batch_mode
        self.rotate_thre = args.cls_rotate_thre
        self.visualize_output = args.visualize_output
        # self.batch_mode = args.cls_batch_mode and (self.batch_num > 1)
        _logger.info(
            "recognize in {} mode {}".format(
                "batch" if self.batch_mode else "serial",
                "batch_size: " + str(self.batch_num) if self.batch_mode else "",
            )
        )

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.cls_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.cls_algorithm in algo_to_model_name, f"Invalid cls_algorithm {args.cls_algorithm}. "
        f"Supported classification algorithms are {list(algo_to_model_name.keys())}."

        model_name = algo_to_model_name[args.cls_algorithm]
        self.model = build_model(model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path)
        self.model.set_train(False)
        _logger.info(
            "Init text direction classification model: {} --> {}. Model weights loaded from {}".format(
                args.cls_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        # NOTE: most process hyper-params should be set optimally for the pick algo.
        self.preprocess = Preprocessor(
            task="cls",
            algo=args.cls_algorithm,
            cls_image_shape=args.cls_image_shape,
            cls_batch_mode=self.batch_mode,
            cls_batch_num=self.batch_num,
        )

        # TODO: try GeneratorDataset to wrap preprocess transform on batch for possible speed-up.
        # if use_ms_dataset: ds = ms.dataset.GeneratorDataset(wrap_preprocess, ) in run_batchwise

        self.postprocess = Postprocessor(task="cls", algo=args.cls_algorithm, label_list=args.cls_label_list)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(self, img_or_path_list: list):
        """
        Run text direction classification serially for input images

                Args:
            img_or_path_list: list of str for img path or np.array for RGB image

                Return:
            list of dict, each contains the follow keys for text direction classification result.
            e.g. [{'texts': 'abc', 'confs': 0.9}, {'texts': 'cd', 'confs': 1.0}]
                - texts: text string
                - confs: prediction confidence
        """

        assert isinstance(
            img_or_path_list, list
        ), "Input for text direction classification must be list of images or image paths."
        _logger.info("num images for cls: ", len(img_or_path_list))
        if self.batch_mode:
            cls_res_all, all_rotated_imgs = self.run_batchwise(img_or_path_list)
        else:
            cls_res_all, all_rotated_imgs = [], []
            for i, img_or_path in enumerate(img_or_path_list):
                cls_res, rotated_imgs = self.run_single(img_or_path, i)
                cls_res_all.append(cls_res)
                all_rotated_imgs.extend(rotated_imgs)

        # TODO: add vis and save function
        return cls_res_all, all_rotated_imgs

    def rotate(self, img_batch, batch_res):
        rotated_img_batch = []
        for i, score in enumerate(batch_res["scores"]):
            tmp_img = img_batch[i]
            if int(batch_res["angles"][i]) != 0 and score > self.rotate_thre:
                tmp_img = np.rot90(tmp_img, k=int(int(batch_res["angles"][i]) / 90))
                _logger.info(f"After text direction classification, image is rotated {batch_res['angles'][i]} degree.")
            rotated_img_batch.append(tmp_img.transpose(1, 2, 0))  # c, h, w --> h, w, c for saving and visualization
        return rotated_img_batch

    def run_batchwise(self, img_or_path_list: list):
        """
        Run text direction classification serially for input images

                Args:
            img_or_path_list: list of str for img path or np.array for RGB image

                Return:
            cls_res: list of tuple, where each tuple is  (text, score) - text direction classification result for
                each input image in order. where text is the predicted text string, score is its confidence score.
                e.g. [('apple', 0.9), ('bike', 1.0)]
        """
        cls_res, all_rotated_imgs = [], []
        num_imgs = len(img_or_path_list)

        for idx in tqdm(range(0, num_imgs, self.batch_num)):  # batch begin index i
            batch_begin = idx
            batch_end = min(idx + self.batch_num, num_imgs)
            # print(f"Cls img idx range: [{batch_begin}, {batch_end})")
            # TODO: set max_wh_ratio to the maximum wh ratio of images in the batch. and update it for resize,
            # TODO: which may improve text direction classification accuracy in batch-mode
            # TODO: especially for long text image. max_wh_ratio=max(max_wh_ratio, img_w / img_h).
            # TODO: The short ones should be scaled with a.r. unchanged and padded to max width in batch.

            # preprocess
            # TODO: run in parallel with multiprocessing
            img_batch = []
            for j in range(batch_begin, batch_end):  # image index j
                data = self.preprocess(img_or_path_list[j])
                img_batch.append(data["image"])  # c,h,w
                if self.visualize_output:
                    fn = os.path.basename(data.get("img_path", f"crop_{j}.png")).split(".")[0]
                    show_imgs(
                        [data["image"]],
                        title=fn + "_cls_preprocessed",
                        mean_rgb=[127.0, 127.0, 127.0],
                        std_rgb=[127.0, 127.0, 127.0],
                        is_chw=True,
                        show=False,
                        save_path=os.path.join(self.vis_dir, fn + "_cls_preproc.png"),
                    )

            img_batch = np.stack(img_batch) if len(img_batch) > 1 else np.expand_dims(img_batch[0], axis=0)
            # infer
            net_pred = self.model(ms.Tensor(img_batch))

            # postprocess
            batch_res = self.postprocess(net_pred)
            img_batch = self.rotate(img_batch, batch_res)

            cls_res.extend(list(zip(batch_res["angles"], batch_res["scores"])))
            all_rotated_imgs.extend(img_batch)

        return cls_res, all_rotated_imgs

    def run_single(self, img_or_path, crop_idx=0):
        """
        Text direction classification inference on a single image
        Args:
            img_or_path: str for image path or np.array for image rgb value

        Return:
            dict with keys:
                - texts (str): preditive text string
                - confs (int): confidence of the prediction
        """
        # preprocess
        data = self.preprocess(img_or_path)

        # visualize preprocess result
        if self.visualize_output:
            # show_imgs([data['image_ori']], is_bgr_img=False, title=f'origin_{i}')
            fn = os.path.basename(data.get("img_path", f"crop_{crop_idx}.png")).split(".")[0]
            show_imgs(
                [data["image"]],
                title=fn + "_cls_preprocessed",
                mean_rgb=[127.0, 127.0, 127.0],
                std_rgb=[127.0, 127.0, 127.0],
                is_chw=True,
                show=False,
                save_path=os.path.join(self.vis_dir, fn + "_cls_preproc.png"),
            )
        print("Origin image shape: ", data["image_ori"].shape)
        print("Preprocessed image shape: ", data["image"].shape)

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_output = self.model(ms.Tensor(net_input))

        # postprocess
        cls_res = self.postprocess(net_output)
        net_input_rot = self.rotate(net_input, cls_res)
        cls_res = (cls_res["angles"][0], cls_res["scores"][0])

        print(f"Crop {crop_idx} cls result:", cls_res)

        return cls_res, net_input_rot


def save_cls_res(cls_res_all, img_paths, include_score=False, save_path="./cls_results.txt"):
    lines = []
    for i, cls_res in enumerate(cls_res_all):
        if include_score:
            img_pred = os.path.basename(img_paths[i]) + "\t" + str(list(cls_res)) + "\n"
        else:
            img_pred = os.path.basename(img_paths[i]) + "\t" + cls_res[0] + "\n"
        lines.append(img_pred)

    with open(save_path, "w") as f:
        f.writelines(lines)

    return lines


if __name__ == "__main__":
    # parse args
    args = parse_args()
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)
    # uncomment it to quick test the infer FPS
    # img_paths = img_paths[:250]

    ms.set_context(mode=args.mode)

    # init classifier
    classifier = DirectionClassifier(args)

    # TODO: warmup

    # run for each image
    start = time()
    cls_res_all, _ = classifier(img_paths)
    t = time() - start
    # save all results in a txt file
    save_fp = os.path.join(save_dir, "cls_results.txt" if args.cls_batch_mode else "cls_results_serial.txt")
    save_cls_res(cls_res_all, img_paths, include_score=True, save_path=save_fp)
    # print('All cls res: ', cls_res_all)
    print("Done! Text direction classification results saved in ", save_dir)
    print("Time cost: ", t, "FPS: ", len(img_paths) / t)
