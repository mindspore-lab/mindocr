## Guideline for Developing Your Transformation

### Writing Guideline

1. Each transformation is a class with a callable function. An example is shown below.

2. The input to the transformation function is always a dict, which contain data info like img_path, raw label, etc. 

3. Please write comments for the __call__ function to clarify the required/modified/added keys in the data dict.

4. Add kwargs in the class init function for extension, which is used to parse global config, such as is_train. 

```python
class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, channel, **kwargs):
        self.is_train = kwargs.get('is_train', True)

    def __call__(self, data: dict):
        '''
        required keys:
            - image
        modified keys:
            - image
        '''
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data
```

### Add Unit Test and Visualization

Please add unit test in `tests/ut/transforms` for the written transformation and try to cover different cases (inputs and settings).

Please visually check the correctness of the transformation on image and annotation using the jupyter notebook. See `transform_tutorial.ipynb`.


### Important Notes
1. For spatial transformation operaions that will be used in text detection inference or evaluation (e.g. determinstic resize, scale), please record the space transformation information in `shape_list`. Otherwise, the postprocessing method won't be able to map the results back to the orignal image space. On how to record `shape_list`, please refer to [DetResize](./det_transforms.py).
