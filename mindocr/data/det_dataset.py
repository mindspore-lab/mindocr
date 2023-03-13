from .base_dataset import BaseDataset

__all__ = ['DetDataset']

class DetDataset(BaseDataset):
    """Data iterator for detection datasets including ICDAR15 dataset. 
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        data_dir, Required
        label_file, Required
        shuffle, Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        is_train
        output_keys (list): indicates the keys in data dict that are expected to output for dataloader. if None, all data keys will be used for return. 

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a subset of the following data. 
            - img_path (str), image path 
            - image (np.array), the format (CHW, RGB) is defined by the transform pipleine 
            - polys (np.array), shape (num_bboxes, num_points, 2)
            - texts (List),   
            - ignore_tags, # 
            - shrink_mask (np.array), binary mask for text region
            - shrink_map (np.array), 
            - threshold_mask (np.array), 
            - threshold_map (np.array), threshold map
        
        You can specify the `output_keys` arg to order the output data for dataloader.

    Notes: 
        1. Dataset file structure should follow:
            data_dir
            ├── images/
            │  │   ├── 000001.jpg
            │  │   ├── 000002.jpg
            │  │   ├── ... 
            ├── annotation_file.txt
        2. Annotation format should follow (img path and annotation are seperated by tab):
            # image path relative to the data_dir\timage annotation information encoded by json.dumps
            ch4_test_images/img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]   
    """
