import logging
import os
import random
import re
import unicodedata
import warnings
from typing import Any, List, Optional

import lmdb
import numpy as np
import PIL
import six

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

__all__ = ["LMDBDataset"]
_logger = logging.getLogger(__name__)


class LMDBDataset(BaseDataset):
    """Data iterator for ocr datasets including ICDAR15 dataset.
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        is_train: whether the dataset is for training
        data_dir: data root directory for lmdb dataset(s)
        shuffle: Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        output_columns (list): optional, indicates the keys in data dict that are expected to output for dataloader.
            if None, all data keys will be used for return.
        filter_max_len (bool): Filter the records where the label is longer than the `max_text_len`.
        extra_count_if_repeat (bool): Count extra length if there is a consectutive pair of same characters,
            e.g., length of "aa" = 3, length of "aab" = 4. It is usually used in CTC alignment, prevent the situation
            if the input has not enough space for the target during alignment.
        max_text_len (int): The maximum text length the dataloader expected. Default: None.
        filter_zero_text_image (bool): Filter the recods where the label does not have any valid character.
            Default: False.
        character_dict_path (str): The path of the character dictionary. It is used to determine if there is any valid
            character in the label. If it is not provided, then `0123456789abcdefghijklmnopqrstuvwxyz` will be used.
            Default: None.
        label_standandize (bool): Apply label standardization (NFKD). default: False.
        random_choice_if_none (bool): Random choose another data if the result returned from data transform is none.
            Default: False.

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item.
        You can specify the `output_columns` arg to order the output data for dataloader.

    Notes:
        1. Dataset file structure should follow:
            data_dir
            ├── dataset01
                ├── data.mdb
                ├── lock.mdb
            ├── dataset02
                ├── data.mdb
                ├── lock.mdb
            ├── ...
    """

    def __init__(
        self,
        is_train: bool = True,
        data_dir: str = "",
        sample_ratio: float = 1.0,
        shuffle: Optional[bool] = None,
        transform_pipeline: Optional[List[dict]] = None,
        output_columns: Optional[List[str]] = None,
        filter_max_len: bool = False,
        extra_count_if_repeat: bool = False,
        max_text_len: Optional[int] = None,
        filter_zero_text_image: bool = False,
        character_dict_path: Optional[str] = None,
        label_standandize: bool = False,
        random_choice_if_none: bool = False,
        check_rec_image: bool = False,
        **kwargs: Any,
    ):
        self.data_dir = data_dir
        self.filter_max_len = filter_max_len
        self.max_text_len = max_text_len
        self.label_standandize = label_standandize
        self.extra_count_if_repeat = extra_count_if_repeat
        self.random_choice_if_none = random_choice_if_none
        self.check_rec_image = check_rec_image

        shuffle = shuffle if shuffle is not None else is_train

        self.lmdb_sets = self.load_list_of_hierarchical_lmdb_dataset(data_dir)
        if len(self.lmdb_sets) == 0:
            raise ValueError(f"Cannot find any lmdb dataset under `{data_dir}`. Please check the data path is correct.")
        self.data_idx_order_list = self.get_dataset_idx_orders(sample_ratio, shuffle)

        # filter the max length
        if filter_max_len:
            if max_text_len is None:
                raise ValueError("`max_text_len` must be provided when `filter_max_len` is True.")
            self.data_idx_order_list = self.filter_idx_list_exceeds_max_length(self.data_idx_order_list)

        # filter zero text image
        if filter_zero_text_image:
            self.data_idx_order_list = self.filter_idx_list_with_zero_text(
                self.data_idx_order_list, character_dict_path
            )

        # create transform
        if transform_pipeline is not None:
            self.transforms = create_transforms(transform_pipeline)
        else:
            raise ValueError("No transform pipeline is specified!")

        self.prefetch(output_columns)

    def prefetch(self, output_columns):
        # prefetch the data keys, to fit GeneratorDataset
        _data = self.data_idx_order_list[0]
        lmdb_idx, file_idx = self.data_idx_order_list[0]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]["txn"], file_idx)
        _data = {"img_lmdb": sample_info[0], "label": sample_info[1]}
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
                        f"Key {k} does not exist in data (available keys: {_data.keys()}). "
                        "Please check the name or the completeness transformation pipeline."
                    )

    @staticmethod
    def count_extra_len_if_repeated(label: str) -> int:
        if len(label) < 2:
            return len(label)
        num = 1
        for i in range(1, len(label)):
            if label[i] == label[i - 1]:
                num += 2
            else:
                num += 1
        return num

    def filter_idx_list_exceeds_max_length(self, idx_list: np.ndarray) -> np.ndarray:
        _logger.info("Start filtering the idx list which exceeds max length...")
        new_idx_list = list()
        for lmdb_idx, file_idx in idx_list:
            label = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]["txn"], int(file_idx), label_only=True)
            if self.extra_count_if_repeat:
                label_length = self.count_extra_len_if_repeated(label)
            else:
                label_length = len(label)

            if label_length > self.max_text_len:
                _logger.warning(
                    f"skip the label with length ({label_length}), "
                    f"which is longer than the max length ({self.max_text_len})."
                )
                continue
            new_idx_list.append((lmdb_idx, file_idx))
        new_idx_list = np.array(new_idx_list)
        return new_idx_list

    def filter_idx_list_with_zero_text(
        self, idx_list: np.ndarray, character_dict_path: Optional[str] = None
    ) -> np.ndarray:
        _logger.info("Start filtering the idx list which has zero text...")
        if character_dict_path is None:
            char_list = list("0123456789abcdefghijklmnopqrstuvwxyz")
            # in case when lower=True, we don't want to filter out the original upper case characters
            char_list = list(char_list) + [x.upper() for x in char_list]
        else:
            char_list = []
            with open(character_dict_path, "r") as f:
                for line in f:
                    c = line.rstrip("\n\r")
                    char_list.append(c)

        char_list = set(char_list)

        new_idx_list = list()
        for lmdb_idx, file_idx in idx_list:
            label = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]["txn"], int(file_idx), label_only=True)

            if len(set(label).intersection(char_list)) == 0:
                _logger.warning(f"skip the label `{label}`, " f"which does not contain any valid character.")
                continue
            new_idx_list.append((lmdb_idx, file_idx))
        new_idx_list = np.array(new_idx_list)
        return new_idx_list

    def load_list_of_hierarchical_lmdb_dataset(self, data_dir):
        if isinstance(data_dir, str):
            results = self.load_hierarchical_lmdb_dataset(data_dir)
        elif isinstance(data_dir, list):
            results = {}
            for sub_data_dir in data_dir:
                start_idx = len(results)
                lmdb_sets = self.load_hierarchical_lmdb_dataset(sub_data_dir, start_idx)
                results.update(lmdb_sets)
        else:
            results = {}

        message = "\n".join(
            [f"{os.path.basename(os.path.abspath(x['rootdir'])):<20}{x['data_size']}" for x in results.values()]
        )
        _logger.info("Number of LMDB records:\n" + message)

        return results

    def load_hierarchical_lmdb_dataset(self, data_dir, start_idx=0):
        lmdb_sets = {}
        dataset_idx = start_idx
        for rootdir, dirs, _ in os.walk(data_dir + "/"):
            if not dirs:
                try:
                    env = lmdb.Environment(
                        rootdir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False
                    )
                except lmdb.Error as e:
                    _logger.warning(str(e))
                    continue
                txn = env.begin(write=False)
                data_size = int(txn.get("num-samples".encode()))
                lmdb_sets[dataset_idx] = {"rootdir": rootdir, "env": env, "txn": txn, "data_size": data_size}
                dataset_idx += 1
        return lmdb_sets

    def get_dataset_idx_orders(self, sample_ratio, shuffle):
        n_lmdbs = len(self.lmdb_sets)
        total_sample_num = 0
        for idx in range(n_lmdbs):
            total_sample_num += self.lmdb_sets[idx]["data_size"]
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for idx in range(n_lmdbs):
            tmp_sample_num = self.lmdb_sets[idx]["data_size"]
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = idx
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num

        if shuffle:
            np.random.shuffle(data_idx_order_list)

        data_idx_order_list = data_idx_order_list[: round(len(data_idx_order_list) * sample_ratio)]

        return data_idx_order_list

    def get_lmdb_sample_info(self, txn, idx, label_only=False):
        label_key = "label-%09d".encode() % idx
        label = txn.get(label_key)
        if label is None:
            raise ValueError(f"Cannot find key {label_key}")
        label = label.decode("utf-8")

        if self.label_standandize:
            label = unicodedata.normalize("NFKD", label)

        if label_only:
            return label

        img_key = "image-%09d".encode() % idx
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]

        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]["txn"], int(file_idx))

        if sample_info is None and self.random_choice_if_none:
            _logger.warning("sample_info is None, randomly choose another data.")
            random_idx = np.random.randint(self.__len__())
            return self.__getitem__(random_idx)

        data = {"img_lmdb": sample_info[0], "label": sample_info[1]}

        if self.check_rec_image:
            if self._check_rec_image(data):
                return self._next_image(idx)

        # perform transformation on data
        try:
            data = run_transforms(data, transforms=self.transforms)
        except Exception as e:
            if self.random_choice_if_none:
                _logger.warning("data is None after transforms, randomly choose another data.")
                random_idx = np.random.randint(self.__len__())
                return self.__getitem__(random_idx)
            else:
                _logger.warning(f"Error occurred during preprocess.\n {e}")
                raise e

        output_tuple = tuple(data[k] for k in self.output_columns)

        return output_tuple

    def __len__(self):
        return self.data_idx_order_list.shape[0]

    def _next_image(self, index):
        next_index = random.randint(0, self.data_idx_order_list.shape[0] - 1)
        # print("next_index:",next_index,"len:",self.lmdb_sets[index]['num_samples'] - 1)
        return self.__getitem__(idx=next_index)

    def _check_rec_image(self, data):
        img_lmdb = data["img_lmdb"]
        label = data["label"]
        label = label.encode("utf-8")
        label = str(label, "utf-8")
        try:
            label = re.sub("[^0-9a-zA-Z]+", "", label)
            buf = six.BytesIO()
            buf.write(img_lmdb)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                image = PIL.Image.open(buf).convert("RGB")
            if not _check_image(image, pixels=6):
                return True
        except Exception:
            return True

        return False


def _check_image(x, pixels=6):
    if x.size[0] <= pixels or x.size[1] <= pixels:
        return False
    else:
        return True
