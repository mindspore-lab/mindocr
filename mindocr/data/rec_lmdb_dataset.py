import os
from typing import Any, List, Optional

import lmdb
import numpy as np

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

__all__ = ["LMDBDataset"]


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
        max_text_len (int): The maximum text length the dataloader expected.

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
        max_text_len: Optional[int] = None,
        **kwargs: Any,
    ):
        self.data_dir = data_dir
        self.filter_max_len = filter_max_len
        self.max_text_len = max_text_len

        shuffle = shuffle if shuffle is not None else is_train

        self.lmdb_sets = self.load_list_of_hierarchical_lmdb_dataset(data_dir)
        if len(self.lmdb_sets) == 0:
            raise ValueError(f"Cannot find any lmdb dataset under `{data_dir}`. Please check the data path is correct.")
        self.data_idx_order_list = self.get_dataset_idx_orders(sample_ratio, shuffle)

        # filter the max length
        if filter_max_len:
            if max_text_len is None:
                raise ValueError("`max_text_len` must be provided when `filter_max_len` is True.")
            self.data_idx_order_list = self.filter_idx_list(self.data_idx_order_list)

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

    def filter_idx_list(self, idx_list: np.ndarray) -> np.ndarray:
        print("Start filtering the idx list...")
        new_idx_list = list()
        for lmdb_idx, file_idx in idx_list:
            label = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]["txn"], int(file_idx), label_only=True)
            if len(label) > self.max_text_len:
                print(
                    f"WARNING: skip the label with length ({len(label)}), "
                    f"which is longer than than max length ({self.max_text_len})."
                )
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

        return results

    def load_hierarchical_lmdb_dataset(self, data_dir, start_idx=0):
        lmdb_sets = {}
        dataset_idx = start_idx
        for rootdir, dirs, _ in os.walk(data_dir + "/"):
            if not dirs:
                env = lmdb.open(rootdir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
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

        if label_only:
            return label

        img_key = "image-%09d".encode() % idx
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]["txn"], int(file_idx))

        data = {"img_lmdb": sample_info[0], "label": sample_info[1]}

        # perform transformation on data
        data = run_transforms(data, transforms=self.transforms)
        output_tuple = tuple(data[k] for k in self.output_columns)

        return output_tuple

    def __len__(self):
        return self.data_idx_order_list.shape[0]
