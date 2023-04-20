import os
from typing import Union, List
from abc import ABC, abstractmethod

__all__ = ['BaseDataset']


class BaseDataset(ABC):
    """
    Base dataset to parse dataset files.

    Args:
        - data_dir:
        - label_file:
        - output_columns (List(str)): names of elements in the output tuple of __getitem__
    Attributes:
        data_list (List(Tuple)): source data items (e.g., containing image path and raw annotation)
    """
    def __init__(self,
                 data_dir: Union[str, List[str]],
                 label_file: Union[str, List[str]] = None,
                 **kwargs):
        self._index = 0
        self.data_list = []

        # check files
        self.data_dir = self._check_existence(data_dir)

        self.label_file = []
        if label_file is not None:
            self.label_file = self._check_existence(label_file)

    @abstractmethod
    def __getitem__(self, index):
        ...

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self):
            item = self[self._index]
            self._index += 1
            return item
        raise StopIteration

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def _check_existence(path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            if not os.path.exists(p):
                raise ValueError(f"{p} does not exist. Please check the yaml file for both train and eval.")
        return path
