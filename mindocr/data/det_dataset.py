import logging
import os
import random
from typing import List, Union

import numpy as np
from scipy.io import loadmat

from .base_dataset import BaseDataset

__all__ = ["DetDataset", "SynthTextDataset"]
_logger = logging.getLogger(__name__)


class DetDataset(BaseDataset):
    """
    General dataset for text detection
    The annotation format should follow:

    .. code-block: none

        # image file name\tannotation info containing text and polygon points encoded by `json.dumps`
        img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]

    Args:
        is_train (bool): whether it is in training stage
        data_dir (str):  directory to the image data
        label_file (Union[str, List[str]]): (list of) path to the label file(s),
            where each line in the label fle contains the image file name and its ocr annotation.
        sample_ratio (Union[float, List[float]]): sample ratios for the data items in label files
        shuffle(bool): shuffle samples within dataset. Useful only when `sample_ratio` is set below 1. (Default: False).
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

    def __init__(
        self,
        data_dir: Union[str, List[str]] = None,
        label_file: Union[List, str] = None,
        sample_ratio: Union[List, float] = 1.0,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, label_file=label_file)

        if isinstance(sample_ratio, float):
            sample_ratio = [sample_ratio] * len(self.label_file)

        # load date file list
        self.data_list = self.load_data_list(self.label_file, sample_ratio, shuffle)
        self.output_columns = ["image", "label"]

    def __getitem__(self, index):
        image = np.fromfile(self.data_list[index]["img_path"], np.uint8)
        return image, self.data_list[index]["label"]

    def load_data_list(
        self, label_file: List[str], sample_ratio: List[float], shuffle: bool = False, **kwargs
    ) -> List[dict]:
        """
        Load data list from label_file which contains information of image paths and annotations
        Args:
            label_file: annotation file path(s)
            sample_ratio: sample ratio for data items in each annotation file
            shuffle: shuffle the data list
        Returns:
            data (List[dict]): A list of annotation dict, which contains keys: img_path, annot...
        """
        # parse image file path and annotation and load
        data_list = []
        for idx, label_fp in enumerate(label_file):
            img_dir = self.data_dir[idx]
            with open(label_fp, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if shuffle:
                    lines = random.sample(lines, round(len(lines) * sample_ratio[idx]))
                else:
                    lines = lines[: round(len(lines) * sample_ratio[idx])]

                for line in lines:
                    img_name, annot_str = self._parse_annotation(line)

                    img_path = os.path.join(img_dir, img_name)
                    assert os.path.exists(img_path), f"{img_path} does not exist!"

                    data = {"img_path": img_path, "label": annot_str}
                    data_list.append(data)

        return data_list

    @staticmethod
    def _parse_annotation(data_line: str):
        data_line = data_line.strip()  # trim leading and trailing whitespace
        if "\t" in data_line:
            img_name, annot_str = data_line.split("\t")
        elif " " in data_line:
            img_name, annot_str = data_line.split(" ")
        else:
            raise ValueError(
                "Incorrect label file format: the file name and the label should be separated by " "a space or tab"
            )

        return img_name, annot_str


class SynthTextDataset(DetDataset):
    def __init__(self, data_dir: Union[str, List[str]] = None, label_file: Union[List, str] = None, **kwargs):
        super().__init__(data_dir, label_file, **kwargs)
        self.output_columns = ["img_path", "polys", "texts", "ignore_tags"]

    def load_data_list(self, label_file: List[str], *args):
        _logger.info("Loading SynthText dataset. It might take a while...")
        mat = loadmat(label_file[0])

        data_list = []
        for image, boxes, texts in zip(mat["imnames"][0], mat["wordBB"][0], mat["txt"][0]):
            texts = [t for text in texts.tolist() for t in text.split()]  # TODO: check the correctness of texts order
            data_list.append(
                {
                    "img_path": os.path.join(self.data_dir[0], image.item()),
                    "polys": boxes.transpose().reshape(-1, 4, 2),  # some labels have (4, 2) shape (no batch dimension)
                    "texts": texts,
                    "ignore_tags": np.array([False] * len(texts)),
                }
            )

        return data_list
