## Postprocessing Module Guideline

### Common Protocols

1. Each postprocessing module is required to be implemented in form of **class** 
2. Use `__call__()` to execute the post-process, which accepts the network prediction and additional data information (optional)  as input, and return postprocessing results (text polygons, recognized text string, scores, etc) in a **dictionary**. 


### Detection Postprocessing API Protocols
1. class  `__init__()` should support the follow args:
    - `box_type` (string): options are ['quad', 'polys'] for quadriateral and polygon text representation.  
    - `rescale_fields` (List[str]=None): indicates which fields in the output dict will be rescaled to the original image space. Options: `polys`, `bboxes`, `beziers` 

2. `__call__()` method: 
    - Input args:
        - `pred` (Union[Tensor, Tuple[Tensor]]): network prediction 
        - `shape_list` (List[float]=None):  [h, w, scale_h, scale_w], scale_h and scale_w are the scale ratio of image height and width in data processing respectively. 
        - `**kwargs`: args for extension

    - Return: det_res as a dictionary with the following keys
        - `polys` (List[np.ndarray]): list of numpy array, list length = num text polygons, array shape is [num_points, 2]. If there is not text polygons, return [[]] 
        - `scores` (List[float]): confidence of each prediction 

3. Notes: 
    - `shape_list` is used to rescale the image and predicted polygons (or other fields) back to orignal image shape for next-step cropping or evaluation. It works with `rescale_fields` to decide whether to do rescaling and which fields are to be rescaled.  
    - `shape_list` will be pared to the postprocessing method during evaluation if `output_columns` in YAML config file contains "shape_list". The "shape_list" key is originally generated in image resize transformation, such as `DetResize` (or other spatial transformations if possible). Example: [psenet config](configs/det/psenet/pse_r152_icdar15.yaml). 
    - You can also directly parse `shape_list` to the postprocessing method without loading a YAML config file in inference. Example: `shape_list = data["shape_list"]`, `my_detprocess(pred, shape_list)`


4. Example code: [PSEPostprocess](mindocr/postprocess/det_postprocess.py)


### Recognition Postprocessing API Protocols

1. class  `__init__()` should support the follow args:
        - character_dict_path
        - use_space_char
        - blank_at_last
        - lower
    Please see the API docs in [RecCTCLabelDecode](mindocr/postprocess/rec_postprocess.py) for argument illustration.  

2. `__call__()` method: 
    - Input args:
        - `pred` (Union[Tensor, Tuple[Tensor]]): network prediction 
        - `**kwargs`: args for extension

    - Return: det_res as a dictionary with the following keys
        - `texts` (List[str]): list of preditected text string 
        - `confs` (List[float]): confidence of each prediction 

Example code: [RecCTCLabelDecode](mindocr/postprocess/rec_postprocess.py) 
