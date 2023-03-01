# MindOCR (under development)
A toolbox of OCR models, algorithms, and pipelines based on MindSpore

**Features:**

- Unified framework to support training text detection and recogintion models


## Quick Start (for dev)

### Data preparation

Download ic15 dataset.

Convert to the required annotation format using `tools/data_converter/convert.py`, referring to `tools/data_converters/README.md`

Change the annotation file path in the yaml file under `configs` accordingly.

### Training 

#### Text Detection Model (DBNet)

``` python
python tools/train.py --config configs/det/db_r50_icdar15.yaml
```

#### Text Recognition Model (CRNN)

``` python
python tools/train.py --config configs/rec/crnn_icdar15.yaml
```


## Build and Test A New Model

### Build your own model
Please follow this [guideline](./mindocr/models/README.md)

### Test model writing

Change the model name and yaml path for your model in `tests/ut/test_models`, e.g.

``` python
    test_model_by_name('dbnet_r50')
    test_model_by_yaml('configs/det/db_r50_icdar15.yaml')
```

Run in the root dir:

``` shell 
python tests/ut/test_models.py
```


