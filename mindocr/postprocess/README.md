## Guideline for Postprocessing Module

### Common Protocols

1. Each postprocessing module is a **class** with a callable function.
2. The input to the postprocessing function is network prediction and additional data information if needed.
3. The output of the postprocessing function is a alwasy a dict, where the key is a field name, such as 'polys' for polygons in text detection, 'text' for text detection.


### Detection Postprocessing API Protocols
1. class naming: Det{Method}Postprocess

2. class  `__init__()` args:
    - `box_type` (string): options are ["quad', 'polys"] for quadriateral and polygon text representation.
    - `rescale_fields` (List[str]='polys'): indicates which fields in the output dict will be rescaled to the original image space. Field name: "polys" for polygons

3. `__call__()` method: If inherit from `DetBasePostprocess `DetBasePostprocess``, you don't need to implement this method in your Postproc. class.
    Execution entry for postprocessing, which postprocess network prediction on the transformed image space to get text boxes (by `self._postprocess()` function) and then rescale them back to the original image space (by `self.rescale()` function).

    - Input args:
        - `pred` (Union[Tensor, Tuple[Tensor]]): network prediction for input batch data, shape [batch_size, ...]
        - `shape_list` (Union[List, np.ndarray, ms.Tensor]): shape and scale info for each image in the batch, shape [batch_size, 4]. Each internal array of length 4 is [src_h, src_w, scale_h, scale_w], where src_h and src_w are height and width of the original image, and scale_h and scale_w are their scale ratio after image resizing respectively.
        - `**kwargs`: args for extension

    - Return: detection result as a dictionary with the following keys
        - `polys` (List[List[np.ndarray]): predicted polygons mapped on the **original** image space, shape [batch_size, num_polygons, num_points, 2]. If `box_type` is 'quad', num_points=4, and the internal np.ndarray is of shape [4, 2]
        - `scores` (List[float]): confidence scores for the predicted polygons, shape (batch_size, num_polygons)

4. `_postprocess()` method: Implement your postprocessing method here if inherit from `DetBasePostprocess`
    Postprocess network prediction to get text boxes on the transformed image space (which will be rescaled back to original image space in __call__ function)

    - Input args:
        - `pred` (Union[Tensor, Tuple[Tensor]]): network prediction for input batch data, shape [batch_size, ...]
        - `**kwargs`: args for extension

    - Return: postprocessing result as a dict with keys:
        - `polys` (List[List[np.ndarray]): predicted polygons on the **transformed** (i.e. resized normally) image space, of shape (batch_size, num_polygons, num_points, 2). If `box_type` is 'quad', num_points=4.
        - `scores` (np.ndarray): confidence scores for the predicted polygons, shape (batch_size, num_polygons)

    - Notes:
        - Please cast `pred` to the type you need in your implementation. Some postprocesssing steps use ops from mindspore.nn and prefer Tensor type, while some steps prefer np.ndarray type required in other libraries.
        - `_postprocess()` should **NOT round** the text box `polys` to integer in return, because they will be recaled and then rounded in the end. Rounding early will cause larger error in polygon rescaling and results in **evaluation performance degradation**, especially on small datasets.

5. About rescaling the polygons back to the original image spcae
    - The rescaling step is necessary for a fair evaluation and is needed in cropping text regions from the orginal image in inference.
    - To enable rescaling for evaluation
        1. add "shape_list" to the `eval.dataset.output_columns` in the YAML config file of the model.
        2. make sure `rescale_fields` is not None (default is ["polys"])
    - To enable rescaling in inference:
        1. directly parse `shape_list` (which is got from data["shape_list"] after data loading) to the postprocessing function.
     It works with `rescale_fields` to decide whether to do rescaling and which fields are to be rescaled.
    - `shape_list` is originally recorded in image resize transformation, such as `DetResize`.


**Example Code:** [DetBasePostprocess](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/det_base_postprocess.py) and [DetDBPostprocess](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/det_db_postprocess.py)


### Recognition Postprocessing API Protocols

1. class  `__init__()` should support the follow args:
        - character_dict_path
        - use_space_char
        - blank_at_last
        - lower
    Please see the API docs in [RecCTCLabelDecode](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/rec_postprocess.py) for argument illustration.

2. `__call__()` method:
    - Input args:
        - `pred` (Union[Tensor, Tuple[Tensor]]): network prediction
        - `**kwargs`: args for extension

    - Return: det_res as a dictionary with the following keys
        - `texts` (List[str]): list of preditected text string
        - `confs` (List[float]): confidence of each prediction

**Example code:** [RecCTCLabelDecode](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/rec_postprocess.py)
