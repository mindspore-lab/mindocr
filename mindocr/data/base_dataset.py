import os
from typing import List, Union

__all__ = ["BaseDataset"]


class BaseDataset(object):
    """
    Base dataset to parse dataset files.

    Args:
        - data_dir:
        - label_file:
        - output_columns (List(str)): names of elements in the output tuple of __getitem__
    Attributes:
        data_list (List(Tuple)): source data items (e.g., containing image path and raw annotation)
    """

    def __init__(
        self,
        data_dir: Union[str, List[str]],
        label_file: Union[str, List[str]] = None,
        output_columns: List[str] = None,
        **kwargs,
    ):
        self._index = 0
        self.data_list = []

        # check files
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        for f in data_dir:
            if not os.path.exists(f):
                raise ValueError(f"data_dir '{f}' does not existed. Please check the yaml file for both train and eval")
        self.data_dir = data_dir

        if label_file is not None:
            if isinstance(label_file, str):
                label_file = [label_file]
            for f in label_file:
                if not os.path.exists(f):
                    raise ValueError(
                        f"label_file '{f}' does not existed. Please check the yaml file for both train and eval"
                    )
        else:
            label_file = []
        self.label_file = label_file

        # must specify output column names
        self.output_columns = output_columns

    def __getitem__(self, index):
        # return self.data_list[index]
        raise NotImplementedError

    def set_output_columns(self, column_names: List[str]):
        self.output_columns = column_names

    def get_output_columns(self) -> List[str]:
        """
        get the column names for the output tuple of __getitem__, required for data mapping in the next step
        """
        # raise NotImplementedError
        return self.output_columns

    def __next__(self):
        if self._index >= len(self.data_list):
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return len(self.data_list)

    def _load_image_bytes(self, img_path):
        """load image bytes (prepared for decoding)"""
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        return image_bytes
