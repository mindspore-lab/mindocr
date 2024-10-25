## 数据模块指南

### 代码结构
``` text
├── README.md
├── __init__.py
├── base_dataset.py  				# base dataset class with __getitem__
├── builder.py					# API for create dataset and loader
├── det_dataset.py				# general text detection dataset class
├── rec_dataset.py				# general rec detection dataset class
├── rec_lmdb_dataset.py				# LMDB dataset class
└── transforms
    ├── det_transforms.py			# processing and augmentation ops (callabel classes) especially for detection tasks
    ├── general_transforms.py			# general processing and augmentation ops (callabel classes)
    ├── modelzoo_transforms.py			# transformations adopted from modelzoo
    ├── rec_transforms.py			# processing and augmentation ops (callabel classes) especially for recognition tasks
    └── transforms_factory.py			# API for create and run transforms
```

### 如何添加自己的dataset类

1. 继承BaseDataset类
2. 在BaseDataset中重写以下文件和标注解析函数。

    def load_data_list(self, label_file: Union[str, List[str]], sample_ratio: Union[float, List] = 1.0,  shuffle: bool = False, **kwargs) -> List[dict]

    def _parse_annotation(self, data_line: str) -> Union[dict, List[dict]]

### 如何添加自己的数据转换

请参考[定制化数据转换开发指导](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/transforms/README_CN.md)
