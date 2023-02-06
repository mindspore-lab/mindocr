# mindocr
A toolbox of OCR models, algorithms, and pipelines based on MindSpore


## Dev and Test

### Dev your model
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


