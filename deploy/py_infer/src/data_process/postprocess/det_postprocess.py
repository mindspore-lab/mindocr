import os
import sys

import mindspore as ms

from .det_east_postprocess import EASTPostprocess
from .det_fce_postprocess import FCEPostprocess
from .det_sast_postprocess import SASTPostprocess

# add mindocr root path, and import postprocess from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess import det_db_postprocess, det_pse_postprocess  # noqa

__all__ = ["DBPostprocess", "EASTPostprocess", "PSEPostprocess", "SASTPostprocess", "FCEPostprocess"]

DBPostprocess = det_db_postprocess.DBPostprocess


class PSEPostprocess(det_pse_postprocess.PSEPostprocess):
    def __init__(
        self,
        binary_thresh=0.5,
        box_thresh=0.85,
        min_area=16,
        box_type="quad",
        scale=4,
        output_score_kernels=False,
        rescale_fields=["polys"],
    ):
        # ascend310/310P doesn't support these actions, need CPU to do the following actions
        ms.set_context(device_target="CPU")
        super().__init__(
            binary_thresh=binary_thresh,
            box_thresh=box_thresh,
            min_area=min_area,
            box_type=box_type,
            scale=scale,
            output_score_kernels=output_score_kernels,
            rescale_fields=rescale_fields,
        )
