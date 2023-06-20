from typing import Dict, List, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Tensor

from ..utils.logger import Logger

__all__ = ["DetBasePostprocess"]
_logger = Logger("mindocr")


class DetBasePostprocess:
    """
    Base class for all text detection postprocessings.

    Args:
        rescale_fields: names of fields to rescale back to the shape of the original image.
        box_type: text region representation type after postprocessing. Options: ['quad', 'poly']
    """

    def __init__(self, rescale_fields: list, box_type: str = "quad"):
        assert box_type in ["quad", "poly"], f"box_type must be `quad` or `poly`, but found {box_type}"

        self._rescale_fields = rescale_fields
        self.warned = False
        if self._rescale_fields is None:
            _logger.warning("`rescale_filed` is None. Cannot rescale the predicted polygons to original image space")

    def _postprocess(self, pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray], **kwargs) -> dict:
        """
        Postprocess network predictions to extract text boxes on the transformed (input to the network) image space
        (which will be rescaled back to original image space in __call__ function)

        Args:
            pred: network prediction for input batch data, shape [batch_size, ...]

        Return:
            postprocessing result as a dict with keys:
                - polys (List[List[np.ndarray]): predicted polygons on the **transformed**
                (i.e. resized normally) image space, of shape (batch_size, num_polygons, num_points, 2).
                If `box_type` is 'quad', num_points=4.
                - scores (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)

        Notes:
            - Please cast `pred` to the type you need in your implementation. Some postprocessing steps use ops from
              mindspore.nn and prefer Tensor type, while some steps prefer np.ndarray type required in other libraries.
            - `_postprocess()` should **NOT round** the text box `polys` to integer in return, because they will be
              rescaled and then rounded in the end. Rounding early will cause larger error in polygon rescaling and
              results in **evaluation performance degradation**, especially on small datasets.
        """
        raise NotImplementedError

    def __call__(
        self,
        pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray],
        shape_list: Union[np.ndarray, ms.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Execution entry for postprocessing, which postprocess network prediction on the transformed image space to get
        text boxes and then rescale them back to the original image space.

        Args:
            pred (Union[Tensor, Tuple[Tensor], np.ndarray]): network prediction for input batch data,
                shape [batch_size, ...]
            shape_list (Union[np.ndarray, ms.Tensor]): shape and scale info for each image in the batch,
                shape [batch_size, 4]. Each internal array is [src_h, src_w, scale_h, scale_w],
                where src_h and src_w are height and width of the original image, and scale_h and scale_w
                are their scale ratio during image resizing.

        Returns:
            detection result as a dict with keys:
                - polys (List[List[np.ndarray]): predicted polygons mapped on the **original** image space,
                  shape [batch_size, num_polygons, num_points, 2]. If `box_type` is 'quad', num_points=4,
                  and the internal np.ndarray is of shape [4, 2]
                - scores (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)
        """

        # 1. Check input type. Covert shape_list to np.ndarray
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()

        if shape_list is not None:
            assert shape_list.shape[0] and shape_list.shape[1] == 4, (
                "The shape of each item in shape_list must be 4: [raw_img_h, raw_img_w, scale_h, scale_w]. "
                f"But got shape_list of shape {shape_list.shape}"
            )
        else:
            # shape_list = [[pred.shape[2], pred.shape[3], 1.0, 1.0] for i in range(pred.shape[0])] # H, W
            # shape_list = np.array(shape_list, dtype='float32')

            _logger.warning(
                "`shape_list` is None in postprocessing. Cannot rescale the prediction result to original "
                "image space, which can lead to inaccurate evaluation. You may add `shape_list` to `output_columns` "
                "list under eval section in yaml config file, or directly parse `shape_list` to postprocess callable "
                "function."
            )
            self.warned = True

        # 2. Core process
        result = self._postprocess(pred, **kwargs)

        # 3. Rescale processing results
        if shape_list is not None and self._rescale_fields is not None:
            result = self.rescale(result, shape_list)

        return result

    @staticmethod
    def _rescale_polygons(polygons: Union[List[np.ndarray], np.ndarray], shape_list: np.ndarray):
        """
        polygons (Union[List[np.ndarray], np.ndarray]): polygons for an image, shape [num_polygons, num_points, 2],
            value: xy coordinates for all polygon points
        shape_list (np.ndarray): shape and scale info for the image, shape [4,], value: [src_h, src_w, scale_h, scale_w]
        """
        scale = shape_list[:1:-1]
        size = shape_list[1::-1] - 1

        if isinstance(polygons, np.ndarray):
            polygons = np.clip(np.round(polygons / scale), 0, size)
        else:  # if polygons have different number of vertices and stored as a list
            polygons = [np.clip(np.round(poly / scale), 0, size) for poly in polygons]

        return polygons

    def rescale(self, result: Dict, shape_list: np.ndarray) -> dict:
        """
        rescale result back to original image shape

        Args:
            result (dict) with keys for the input data batch
                polys (np.ndarray): polygons for a batch of images, shape [batch_size, num_polygons, num_points, 2].
            shape_list (np.ndarray): image shape and scale info, shape [batch_size, 4]

        Return:
            rescaled result specified by rescale_field
        """

        for field in self._rescale_fields:
            assert (
                field in result
            ), f"Invalid field {field}. Found fields in intermediate postprocess result are {list(result.keys())}"

            for i, sample in enumerate(result[field]):
                if len(sample) > 0:
                    result[field][i] = self._rescale_polygons(sample, shape_list[i])

        return result
