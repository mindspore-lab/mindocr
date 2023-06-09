import os
import random
from typing import List, Union

import numpy as np
from scipy.io import loadmat

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

__all__ = ["DetDataset", "SynthTextDataset"]


class DetDataset(BaseDataset):
    """
    General dataset for text detection
    The annotation format should follow:

    .. code-block: none

        # image file name\tannotation info containing text and polygon points encoded by json.dumps
        img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]

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

    def __init__(
        self,
        is_train: bool = True,
        data_dir: Union[str, List[str]] = None,
        label_file: Union[List, str] = None,
        sample_ratio: Union[List, float] = 1.0,
        shuffle: bool = None,
        transform_pipeline: List[dict] = None,
        output_columns: List[str] = None,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, label_file=label_file, output_columns=output_columns)

        # check args
        if isinstance(sample_ratio, float):
            sample_ratio = [sample_ratio] * len(self.label_file)

        shuffle = shuffle if shuffle is not None else is_train

        # load date file list
        self.data_list = self.load_data_list(self.label_file, sample_ratio, shuffle)

        # create transform
        if transform_pipeline is not None:
            global_config = dict(is_train=is_train)
            self.transforms = create_transforms(transform_pipeline, global_config)
        else:
            raise ValueError("No transform pipeline is specified!")

        # prefetch the data keys, to fit GeneratorDataset
        _data = self.data_list[0].copy()  # WARNING: shallow copy. Do deep copy if necessary.
        _data = run_transforms(_data, transforms=self.transforms)
        _available_keys = list(_data.keys())

        if output_columns is None:
            self.output_columns = _available_keys
        else:
            self.output_columns = []
            for k in output_columns:
                if k in _data:
                    self.output_columns.append(k)
                else:
                    raise ValueError(
                        f"Key '{k}' does not exist in data (available keys: {_data.keys()}). "
                        "Please check the name or the completeness transformation pipeline."
                    )

    def __getitem__(self, index):
        data = self.data_list[index].copy()  # WARNING: shallow copy. Do deep copy if necessary.

        # perform transformation on data
        try:
            data = run_transforms(data, transforms=self.transforms)
            output_tuple = tuple(data[k] for k in self.output_columns)
        except Exception as e:
            print(f"Error occurred while processing the image: {self.data_list[index]['img_path']}\n", e, flush=True)
            return self[random.randrange(len(self.data_list))]  # return another random sample instead

        return output_tuple

    def load_data_list(
        self, label_file: List[str], sample_ratio: List[float], shuffle: bool = False, **kwargs
    ) -> List[dict]:
        """Load data list from label_file which contains infomation of image paths and annotations
        Args:
            label_file: annotation file path(s)
            sample_ratio sample ratio for data items in each annotation file
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
                    if annot_str == "[]":
                        continue
                    img_path = os.path.join(img_dir, img_name)
                    assert os.path.exists(img_path), "{} does not exist!".format(img_path)

                    data = {"img_path": img_path, "label": annot_str}
                    data_list.append(data)

        return data_list

    def _parse_annotation(self, data_line: str):
        data_line_tmp = data_line.strip()
        if "\t" in data_line_tmp:
            img_name, annot_str = data_line.strip().split("\t")
        elif " " in data_line_tmp:
            img_name, annot_str = data_line.strip().split(" ")
        else:
            raise ValueError(
                "Incorrect label file format: the file name and the label should be separated by " "a space or tab"
            )

        return img_name, annot_str


class SynthTextDataset(DetDataset):
    def load_data_list(self, *args):
        print("Loading SynthText dataset. It might take a while...")
        mat = loadmat(os.path.join(self.data_dir[0], "gt.mat"))

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
