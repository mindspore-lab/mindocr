import os
from typing import Any, Optional

import lmdb
import numpy as np

from .base_dataset import BaseDataset

__all__ = ["LMDBDataset"]


class LMDBDataset(BaseDataset):
    """Data iterator for ocr datasets including ICDAR15 dataset.
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        is_train: Whether the dataset is for training. Default: True.
        data_dir: data root directory for lmdb dataset(s). Default: ".".
        sample_ratio: Sampling ratio from current dataset. Default: 1.0.
        shuffle: Optional, if not given, shuffle = is_train. Default: None.
        filter_max_len (bool): Filter the records where the label is longer than the `max_text_len`. Default: False.
        max_text_len (int): The maximum text length the dataloader expected. Default: None.
        **kwargs: Dummmy arguments for compatibilities only.

    Returns:
        data (tuple): Return the tuple of the encoded image array and the corresponding label

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
        data_dir: str = ".",
        sample_ratio: float = 1.0,
        shuffle: Optional[bool] = None,
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

        self.output_columns = ["image", "label"]

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
                try:
                    env = lmdb.Environment(
                        rootdir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False
                    )
                except lmdb.Error as e:  # handle the empty folder
                    print("WARNING: ", str(e))
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

        if label_only:
            return label

        img_key = "image-%09d".encode() % idx
        imgbuf = txn.get(img_key)
        image = np.frombuffer(imgbuf, np.uint8)
        return image, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]["txn"], int(file_idx))
        return sample_info

    def __len__(self):
        return self.data_idx_order_list.shape[0]
