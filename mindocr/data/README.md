## Guideline for Data Module

### Code Structure
``` text
├── README.md
├── __init__.py
├── base_dataset.py  				# base dataset class with __getitem__
├── builder.py					# API for create dataset and loader
├── det_dataset.py				# general text detection dataset class
├── rec_dataset.py				# general rec detection dataset class
├── rec_lmdb_dataset.py				# LMDB dataset class (To be impl.)
└── transforms
    ├── det_transforms.py			# processing and augmentation ops (callabel classes) especially for detection tasks
    ├── general_transforms.py			# general processing and augmentation ops (callabel classes)
    ├── modelzoo_transforms.py			# transformations adopted from modelzoo
    ├── rec_transforms.py			# processing and augmentation ops (callabel classes) especially for recognition tasks
    └── transforms_factory.py			# API for create and run transforms
```

### How to add your own dataset class

1. Inherit from BaseDataset class

2. Rewrite the following file and annotation parsing functions in BaseDataset.

    def load_data_list(self, label_file: Union[str, List[str]], sample_ratio: Union[float, List] = 1.0,  shuffle: bool = False, **kwargs) -> List[dict]

    def _parse_annotation(self, data_line: str) -> Union[dict, List[dict]]

### How to add your own data transformation

Please refer to [Guideline for Developing Your Transformation](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/transforms/README.md)
