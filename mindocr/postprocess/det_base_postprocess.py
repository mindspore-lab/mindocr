from typing import Tuple, Union, List, Dict
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from shapely.geometry import Polygon

from ..data.transforms.det_transforms import expand_poly

__all__ = ["DBBasePostprocess"]


class DetBasePostprocess:
    """
    Base class for all text detection postprocessings.

    Args:
        box_type (str): text region representation type after postprocessing, options: ['quad', 'poly']
        rescale_fields (list): names of fields to rescale back to the shape of the original image.
    """

    def __init__(self, box_type="quad", rescale_fields: List[str] = ["polys"]):
        assert box_type in ["quad", "poly",], f"box_type must be `quad` or `poly`, but found {box_type}"

        self._rescale_fields = rescale_fields
        self.warned = False
        if self._rescale_fields is None:
            print(
                "WARNING: `rescale_filed` is None. Cannot rescale the predicted polygons to original image space"
            )


    def _postprocess(
        self, pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray], **kwargs
    ) -> dict:
        '''
        Postprocess network prediction to get text boxes on the transformed image space (which will be rescaled back to original image space in __call__ function)

        Args:
            pred: network prediction for input batch data, shape [batch_size, ...]

        Return:
            postprocessing result as a dict with keys:
                - polys (List[List[np.ndarray]): predicted polygons on the **transformed** (i.e. resized normally) image space, of shape (batch_size, num_polygons, num_points, 2). If `box_type` is 'quad', num_points=4.
                - scores (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)

        Notes:
            - Please cast `pred` to the type you need in your implementation. Some postprocesssing steps use ops from mindspore.nn and prefer Tensor type, while some steps prefer np.ndarray type required in other libraries.
            - `_postprocess()` should **NOT round** the text box `polys` to integer in return, because they will be recaled and then rounded in the end. Rounding early will cause larger error in polygon rescaling and results in **evaluation performance degradation**, especially on small datasets.
        '''
        raise NotImplementedError


    def __call__(
        self,
        pred: Union[ms.Tensor, Tuple[ms.Tensor], np.ndarray],
        shape_list: Union[List, np.ndarray, ms.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Execution entry for postprocessing, which postprocess network prediction on the transformed image space to get text boxes and then rescale them back to the original image space.

        Args:
            pred (Union[Tensor, Tuple[Tensor], np.ndarray]): network prediction for input batch data, shape [batch_size, ...]
            shape_list (Union[List, np.ndarray, ms.Tensor]): shape and scale info for each image in the batch, shape [batch_size, 4]. Each internal array is [src_h, src_w, scale_h, scale_w], where src_h and src_w are height and width of the original image, and scale_h and scale_w are their scale ratio during image resizing.

        Returns:
            detection result as a dict with keys:
                - polys (List[List[np.ndarray]): predicted polygons mapped on the **original** image space, shape [batch_size, num_polygons, num_points, 2]. If `box_type` is 'quad', num_points=4, and the internal np.ndarray is of shape [4, 2]
                - scores (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)
        """

        # 1. Check input type. Covert shape_list to np.ndarray
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()
        elif isinstance(shape_list, List):
            shape_list = np.array(shape_list, dtype="float32")

        if shape_list is not None:
            assert (
                len(shape_list) > 0 and len(shape_list[0]) == 4
            ), f"The length of each element in shape_list must be 4 for [raw_img_h, raw_img_w, scale_h, scale_w]. But get shape list {shape_list}"
        else:
            # shape_list = [[pred.shape[2], pred.shape[3], 1.0, 1.0] for i in range(pred.shape[0])] # H, W
            # shape_list = np.array(shape_list, dtype='float32')

            print(
                "WARNING: `shape_list` is None in postprocessing. Cannot rescale the prediction result to original image space, which can lead to inaccraute evaluatoin. You may add `shape_list` to `output_columns` list under eval section in yaml config file, or directly parse `shape_list` to postprocess callable function."
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
        polygons (Union[List[np.ndarray], np.ndarray]): polygons for an image, shape [num_polygons, num_points, 2], value: xy coordinates for all polygon points
        shape_list (np.ndarray): shape and scale info for the image, shape [4,], value: [src_h, src_w, scale_h, scale_w]
        """
        scale_w_h = shape_list[:1:-1]
        src_h, src_w = shape_list[0], shape_list[1]

        #print('DEBUG: before rescale, poly 0: ', polygons[0], 'shape list: ', shape_list[0])
        if isinstance(polygons, np.ndarray):
            polygons = np.round(polygons / scale_w_h)
            polygons[...,0] = np.clip(polygons[..., 0], 0, src_w - 1)
            polygons[...,1] = np.clip(polygons[..., 1], 0, src_h - 1)
        else:
            polygons = [np.round(poly / scale_w_h) for poly in polygons]
            for i, poly in enumerate(polygons):
                polygons[i][:,0] = np.clip(poly[:, 0], 0, src_w - 1)
                polygons[i][:,1] = np.clip(poly[:, 1], 0, src_h - 1)

        #print('DEBUG: After rescale, poly 0: ', polygons[0])

        return polygons

    def rescale(self, result: Dict, shape_list: np.ndarray) -> dict:
        """
        rescale result back to orginal image shape

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
            ), f"Invalid field {field}. Found fields in intermidate postprocess result are {list(result.keys())}"
            for i, sample in enumerate(result[field]):
                if len(sample) > 0:
                    result[field][i] = self._rescale_polygons(sample, shape_list[i])

        return result

