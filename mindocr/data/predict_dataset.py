'''
Inference dataset class
'''
import os
import random
from typing import Union, List

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

__all__ = ['PredictDataset']


class PredictDataset(BaseDataset):
    """
        Notes:
        1. The data file structure should be like
            ├── img_dir
            │     ├── 000001.jpg
            │     ├── 000002.jpg
            │     ├── {image_file_name}
    """
    def __init__(self,
                 # is_train: bool = False,
                 dataset_root: str = '',
                 data_dir: str = '',
                 sample_ratio: Union[List, float] = 1.0,
                 shuffle: bool = None,
                 transform_pipeline: List[dict] = None,
                 output_columns: List[str] = None,
                 **kwargs):
        img_dir = os.path.join(dataset_root, data_dir)
        super().__init__(data_dir=img_dir, label_file=None, output_columns=output_columns)
        self.data_list = self.load_data_list(img_dir, sample_ratio, shuffle)
    
        # create transform
        if transform_pipeline is not None:
            self.transforms = create_transforms(transform_pipeline)  # , global_config=global_config)
        else:
            raise ValueError('No transform pipeline is specified!')
    
        # prefetch the data keys, to fit GeneratorDataset
        _data = self.data_list[0]
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
                    raise ValueError(f"Key '{k}' does not exist in data (available keys: {_data.keys()}). "
                                     "Please check the name or the completeness transformation pipeline.")

    def __getitem__(self, index):
        data = self.data_list[index]

        # perform transformation on data
        data = run_transforms(data, transforms=self.transforms)
        output_tuple = tuple(data[k] for k in self.output_columns)

        return output_tuple

    def load_data_list(self,
                       img_dir: str,
                       sample_ratio: List[float],
                       shuffle: bool = False,
                       **kwargs) -> List[dict]:
        # read image file name
        img_filenames = os.listdir(img_dir)
        if shuffle:
            img_filenames = random.sample(img_filenames, round(len(img_filenames) * sample_ratio))
        else:
            img_filenames = img_filenames[:round(len(img_filenames) * sample_ratio)]

        img_paths = [{'img_path': os.path.join(img_dir, filename)} for filename in img_filenames]

        return img_paths