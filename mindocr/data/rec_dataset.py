from .base_dataset import BaseDataset

class RecDataset(BaseDataset):
    """Data iterator for recogition datasets including ICDAR15 dataset.
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        data_dir, Required
        label_files, Required
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
            - text (str), groundtruth text
            - label (List[int]), text string to character indices
            - length (int): the length of text (used to mask the label padded to a max length)

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
