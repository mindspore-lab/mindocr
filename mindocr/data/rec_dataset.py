from .det_dataset import DetDataset

__all__ = ["RecDataset"]


class RecDataset(DetDataset):
    """
    General dataset for text recognition
    The annotation format should follow:

    .. code-block: none

        # image file name\tground truth text
        word_18.png\tSTAGE
        word_19.png\tHarbourFront

    Args:
        is_train (bool): whether it is in training stage
        data_dir (str):  directory to the image data
        label_file (Union[str, List[str]]): (list of) path to the label file(s),
            where each line in the label fle contains the image file name and its ocr annotation.
        sample_ratio (Union[float, List[float]]): sample ratios for the data items in label files
        shuffle(bool): Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                    if None, default transform pipeline for text detection will be taken.
        output_columns (list): required, indicates the keys in data dict that are expected to output for dataloader.
                            if None, all data keys will be used for return.
        global_config: additional info, used in data transformation, possible keys:
            - character_dict_path

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item.
        You can specify the `output_columns` arg to order the output data for dataloader.

    Notes:
        1. The data file structure should be like
            ├── data_dir
            │     ├── 000001.jpg
            │     ├── 000002.jpg
            │     ├── {image_file_name}
            ├── label_file.txt
    """
