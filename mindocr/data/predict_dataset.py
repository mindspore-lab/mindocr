import os
import random
from typing import List, Union

import numpy as np

from .base_dataset import BaseDataset

__all__ = ["PredictDataset"]


class PredictDataset(BaseDataset):
    """
    General dataset for parallel online prediction pipeline

    Args:
        data_dir (str):  directory to the image data
        sample_ratio (Union[float, List[float]]): sample ratios for the data items in label files
        shuffle(bool): shuffle samples within dataset. Useful only when `sample_ratio` is set below 1. (Default: False).

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item.
        You can specify the `output_columns` arg to order the output data for dataloader.

    Notes:
        1. The data file structure should be like
            ├── data_dir
            │     ├── 000001.jpg
            │     ├── 000002.jpg
            │     ├── {image_file_name}
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]] = None,
        sample_ratio: Union[List, float] = 1.0,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir)

        if isinstance(sample_ratio, float):
            sample_ratio = [sample_ratio] * len(self.data_dir)

        # load date file list
        self.data_list = self.load_data_list(sample_ratio, shuffle)
        self.output_columns = ["image", "img_path"]

    def __getitem__(self, index):
        image = np.fromfile(self.data_list[index]["img_path"], np.uint8)
        return image, self.data_list[index]["img_path"]

    def load_data_list(self, sample_ratio: List[float], shuffle: bool = False, **kwargs) -> List[dict]:
        """
        Load data list from label_file which contains information of image paths and annotations
        Args:
            sample_ratio: sample ratio for data items in each annotation file
            shuffle: shuffle the data list
        Returns:
            data (List[dict]): A list of dict, which contains key: img_path
        """
        # parse image file path
        img_paths = []
        for idx, d in enumerate(self.data_dir):
            img_filenames = os.listdir(d)
            if shuffle:
                img_filenames = random.sample(img_filenames, round(len(img_filenames) * sample_ratio[idx]))
            else:
                img_filenames = img_filenames[: round(len(img_filenames) * sample_ratio[idx])]
            img_path = [{"img_path": os.path.join(d, filename)} for filename in img_filenames]
            img_paths.extend(img_path)

        return img_paths
