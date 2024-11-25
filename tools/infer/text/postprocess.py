import os
import sys

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr import build_postprocess


class Postprocessor(object):
    def __init__(self, task="det", algo="DB", rec_char_dict_path=None, **kwargs):
        # algo = algo.lower()
        if task == "det":
            if algo.startswith("DB"):
                if algo == "DB_PPOCRv3":
                    postproc_cfg = dict(
                        name="DBPostprocess",
                        box_type="quad",
                        binary_thresh=0.3,
                        box_thresh=0.7,
                        max_candidates=1000,
                        expand_ratio=1.5,
                    )
                else:
                    postproc_cfg = dict(
                        name="DBPostprocess",
                        box_type="quad",
                        binary_thresh=0.3,
                        box_thresh=0.6,
                        max_candidates=1000,
                        expand_ratio=1.5,
                    )
            elif algo.startswith("PSE"):
                postproc_cfg = dict(
                    name="PSEPostprocess",
                    box_type="quad",
                    binary_thresh=0.0,
                    box_thresh=0.85,
                    min_area=16,
                    scale=1,
                )
            else:
                raise ValueError(f"No postprocess config defined for {algo}. Please check the algorithm name.")
            self.rescale_internally = True
            self.round = True
        elif task == "rec":
            rec_char_dict_path = (
                rec_char_dict_path or "mindocr/utils/dict/ch_dict.txt"
                if algo in ["CRNN_CH", "SVTR_PPOCRv3_CH"]
                else rec_char_dict_path
            )
            # TODO: update character_dict_path and use_space_char after CRNN trained using en_dict.txt released
            if algo.startswith("CRNN") or algo.startswith("SVTR"):
                # TODO: allow users to input char dict path
                if algo == "SVTR_PPOCRv3_CH":
                    postproc_cfg = dict(
                        name="CTCLabelDecode",
                        character_dict_path=rec_char_dict_path,
                        use_space_char=True,
                    )
                else:
                    postproc_cfg = dict(
                        name="RecCTCLabelDecode",
                        character_dict_path=rec_char_dict_path,
                        use_space_char=False,
                    )
            elif algo.startswith("RARE"):
                rec_char_dict_path = (
                    rec_char_dict_path or "mindocr/utils/dict/ch_dict.txt" if algo == "RARE_CH" else rec_char_dict_path
                )
                postproc_cfg = dict(
                    name="RecAttnLabelDecode",
                    character_dict_path=rec_char_dict_path,
                    use_space_char=False,
                )

            else:
                raise ValueError(f"No postprocess config defined for {algo}. Please check the algorithm name.")
        elif task == "ser":
            class_path = "mindocr/utils/dict/class_list_xfun.txt"
            postproc_cfg = dict(name="VQASerTokenLayoutLMPostProcess", class_path=class_path)
        elif task == "layout":
            if algo == "LAYOUTLMV3":
                postproc_cfg = dict(
                    name="Layoutlmv3Postprocess",
                    conf_thres=0.05,
                    iou_thres=0.5,
                    conf_free=False,
                    multi_label=True,
                    time_limit=100,
                )
            elif algo == "YOLOv8":
                postproc_cfg = dict(name="YOLOv8Postprocess", conf_thres=0.5, iou_thres=0.7, conf_free=True)
            else:
                raise ValueError(f"No postprocess config defined for {algo}. Please check the algorithm name.")
        elif task == "table":
            table_char_dict_path = kwargs.get(
                "table_char_dict_path", "mindocr/utils/dict/table_master_structure_dict.txt"
            )
            postproc_cfg = dict(
                name="TableMasterLabelDecode",
                character_dict_path=table_char_dict_path,
                merge_no_span_structure=True,
                box_shape="pad",
            )

        postproc_cfg.update(kwargs)
        self.task = task
        self.postprocess = build_postprocess(postproc_cfg)

    def __call__(self, pred, data=None, **kwargs):
        """
        Args:
            pred: network prediction
            data: (optional)
                preprocessed data, dict, which contains key `shape`
                    - shape: its values are [ori_img_h, ori_img_w, scale_h, scale_w]. scale_h, scale_w are needed to
                      map the predicted polygons back to the orignal image shape.

        return:
            det_res: dict, elements:
                    - polys: shape [num_polys, num_points, 2], point coordinate definition: width (horizontal),
                      height(vertical)
        """

        if self.task == "det":
            if self.rescale_internally:
                shape_list = np.array(data["shape_list"], dtype="float32")
                shape_list = np.expand_dims(shape_list, axis=0)
            else:
                shape_list = None

            output = self.postprocess(pred, shape_list=shape_list)

            if isinstance(output, dict):
                polys = output["polys"][0]
                scores = output["scores"][0]
            else:
                polys, scores = output[0]

            if not self.rescale_internally:
                scale_h, scale_w = data["shape_list"][2:]
                if len(polys) > 0:
                    if not isinstance(polys, list):
                        polys[:, :, 0] = polys[:, :, 0] / scale_w
                        polys[:, :, 1] = polys[:, :, 1] / scale_h
                        if self.round:
                            polys = np.round(polys)
                    else:
                        for i, poly in enumerate(polys):
                            polys[i][:, 0] = polys[i][:, 0] / scale_w
                            polys[i][:, 1] = polys[i][:, 1] / scale_h
                            if self.round:
                                polys[i] = np.round(polys[i])

            det_res = dict(polys=polys, scores=scores)

            return det_res
        elif self.task == "rec":
            output = self.postprocess(pred)
            return output
        elif self.task == "ser":
            output = self.postprocess(
                pred, segment_offset_ids=kwargs.get("segment_offset_ids"), ocr_infos=kwargs.get("ocr_infos")
            )
            return output
        elif self.task == "table":
            output = self.postprocess(pred, labels=kwargs.get("labels"))
            return output
        elif self.task == "layout":
            output = self.postprocess(pred, img_shape=kwargs.get("img_shape"), meta_info=kwargs.get("meta_info"))
            return output
