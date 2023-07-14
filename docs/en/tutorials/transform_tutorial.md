# Transformation Tutorial

[![Download Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindocr/tutorials/transform_tutorial.ipynb)&emsp;

## Mechanism

1. There are 2 types of supported transforms by MindOCR: native MindSpore (such as `Decode`, `HWC2CHW`, etc.) and custom
python transforms. MindOCR provides numerous python transforms suitable for wide variety applications in OCR.
2. Python transform class must have a `__call__` method with a dictionary as input to and output from it as in the
following example:


```python
class RandomHorizontalFlip:
    def __init__(self, polygons: bool = True, p: float = 0.5, **kwargs):
        self._p = p
        self.output_columns = ["image", "polys"] if polygons else ["image"]

    def __call__(self, data: dict) -> dict:
        if random.random() < self._p:
            data["image"] = cv2.flip(data["image"], 1)

            if "polys" in self.output_columns:
                mat = np.float32([[-1, 0, data["image"].shape[1] - 1], [0, 1, 0]])
                data["polys"] = cv2.transform(data["polys"], mat)
                # TODO: assign a new starting point located in the top left
                data["polys"] = data["polys"][:, ::-1, :]  # preserve the original order (e.g. clockwise)

        return data
```

3. The input / output dictionary contains data keys such as `image`, `polys`, etc. A transformation can modify / add new
values to the data dictionary, but should not delete keys (data) from it.
4. Each transformation class must have a `self.output_columns` member. It is necessary for the pipeline data mapping.
5. :warning: For better transformation pipeline efficiency, python and MindSpore transformations should be grouped together in the config files in such way that the number of groups they form is minimized.
That is, python operations should be followed by python operations and MindSpore operations by MindSpore operations. For example:

```yaml
transform_pipeline:
  - Decode:                                           <- MindSpore transformation
  - DetLabelEncode:                                   ─┐
  - ShrinkBinaryMap:                                   │
      min_text_size: 8                                 │
      shrink_ratio: 0.4                                │ Python transformations
  - BorderMap:                                         │
      shrink_ratio: 0.4                                │
      thresh_min: 0.3                                  │
      thresh_max: 0.7                                 ─┘
  - RandomColorAdjust:                                ─┐
      brightness: 0.1255                               │
      saturation: 0.5                                  │
  - Normalize:                                         │ MindSpore transformations
      mean: [ 123.675, 116.28, 103.53 ]                │
      std: [ 58.395, 57.12, 57.375 ]                   │
  - HWC2CHW:                                          ─┘
```
In the example above, the minimum possible number of transformation groups is 3 (we must `Decode` an image first before applying transformations to it).

Available transformations can be found in `mindocr/data/transforms/*_transform.py`

```shell
# import and check available transforms
>>> from mindocr.data.transforms import general_transforms, det_transforms, rec_transforms
>>> general_transforms.__all__
['RandomScale', 'RandomRotate', 'RandomHorizontalFlip']
>>> det_transforms.__all__
['DetLabelEncode', 'BorderMap', 'ShrinkBinaryMap', 'expand_poly', 'PSEGtDecode', 'ValidatePolygons', 'RandomCropWithBBox', 'RandomCropWithMask', 'DetResize']
>>> rec_transforms.__all__
['RecCTCLabelEncode', 'RecAttnLabelEncode', 'RecMasterLabelEncode', 'VisionLANLabelEncode', 'RecResizeImg', 'RecResizeNormForInfer', 'SVTRRecResizeImg', 'Rotate90IfVertical', 'ClsLabelEncode', 'SARLabelEncode', 'RobustScannerRecResizeImg']
```


## Text detection

### 1. Load image and annotations

#### Preparation


```python
%load_ext autoreload
%autoreload 2
%reload_ext autoreload
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
import os

# load the label file which has the info of image path and annotation.
# This file is generated from the ic15 annotations using the converter script.
label_fp = '/Users/Samit/Data/datasets/ic15/det/train/train_icdar2015_label.txt'
root_dir = '/Users/Samit/Data/datasets/ic15/det/train'

data_lines = []
with open(label_fp, 'r') as f:
    for line in f:
        data_lines.append(line)

# just pick one image and its annotation
idx = 3
img_path, annot = data_lines[idx].strip().split('\t')

img_path = os.path.join(root_dir, img_path)
print('img_path', img_path)
print('raw annotation: ', annot)


```

    img_path /Users/Samit/Data/datasets/ic15/det/train/ch4_training_images/img_612.jpg
    raw annotation:  [{"transcription": "where", "points": [[483, 197], [529, 174], [530, 197], [485, 221]]}, {"transcription": "people", "points": [[531, 168], [607, 136], [608, 166], [532, 198]]}, {"transcription": "meet", "points": [[613, 128], [691, 100], [691, 131], [613, 160]]}, {"transcription": "###", "points": [[695, 299], [888, 315], [931, 635], [737, 618]]}, {"transcription": "###", "points": [[709, 19], [876, 8], [880, 286], [713, 296]]}, {"transcription": "###", "points": [[530, 270], [660, 246], [661, 300], [532, 324]]}, {"transcription": "###", "points": [[113, 356], [181, 359], [180, 387], [112, 385]]}, {"transcription": "###", "points": [[281, 328], [369, 338], [366, 361], [279, 351]]}, {"transcription": "###", "points": [[66, 314], [183, 313], [183, 328], [68, 330]]}]


#### MindSpore transform: Decode an image


```python
import numpy as np
from mindspore.dataset.vision import Decode
#img_path = '/Users/Samit/Data/datasets/ic15/det/train/ch4_training_images/img_1.jpg'
decode_image = Decode()

# TODO: check the input keys and output keys for the trans. func.

img_buffer = np.fromfile(img_path, np.uint8)
img  = decode_image(img_buffer)

# visualize
from mindocr.utils.visualize import show_img, show_imgs
show_img(img)
```


![output_13_0](https://user-images.githubusercontent.com/20376974/228160967-262e9fe3-1118-49b2-b269-156e44761edf.png)



```python
import time

start = time.time()
att = 100
for i in range(att):
    img  = decode_image(data)['image']
avg = (time.time() - start) / att

print('avg reading time: ', avg)
```

    avg reading time:  0.004545390605926514


#### Python transform: DetLabelEncode


```python
data['label'] = annot

decode_image = det_transforms.DetLabelEncode()
data = decode_image(data)

#print(data['polys'])
print(data['texts'])

# visualize
from mindocr.utils.visualize import draw_boxes

res = draw_boxes(data['image'], data['polys'])
show_img(res)

```

    ['where', 'people', 'meet', '###', '###', '###', '###', '###', '###']



![output_16_1](https://user-images.githubusercontent.com/20376974/228161131-c11209d1-f3f0-4a8c-a763-b72d729a4084.png)


### 2. Image and annotation processing/augmentation

#### RandomCropWithBBox


```python
from mindocr.data.transforms.det_transforms import RandomCropWithBBox
import copy

#crop_data = det_transforms.EastRandomCropData(size=(640, 640))
crop_data = RandomCropWithBBox(crop_size=(640, 640))

show_img(data['image'])
for i in range(2):
    data_cache = copy.deepcopy(data)
    data_cropped = crop_data(data_cache)

    res_crop = draw_boxes(data_cropped['image'], data_cropped['polys'])
    show_img(res_crop)
```


![output_19_0](https://user-images.githubusercontent.com/20376974/228161220-c56ebd8d-37a0-48a8-9746-3c8da0eaddbb.png)



![output_19_1](https://user-images.githubusercontent.com/20376974/228161306-8359d0b5-f77d-4ec6-8192-fecdaa4c8a1e.png)



![output_19_2](https://user-images.githubusercontent.com/20376974/228161334-8232f0ac-7ca0-49d6-b15a-45b58cb80003.png)


#### ColorJitter


```python
from mindspore.dataset.vision import RandomColorAdjust
random_color_adj = RandomColorAdjust(brightness=0.4, saturation=0.5)

data_cache = copy.deepcopy(data)
#data_cache['image'] = data_cache['image'][:,:, ::-1]
data_adj = random_color_adj(data_cache)
#print(data_adj)
show_img(data_adj['image'], is_bgr_img=True)
```


![output_21_0](https://user-images.githubusercontent.com/20376974/228161397-c64faae6-b4a2-41ff-9531-5bced781fd9d.png)


## FAQ
1. **During execution, I see the following warning: `Using shared memory queue, but rowsize is larger than allocated
memory max_rowsize: X MB, current rowsize: X MB`. How can I fix this?**</br>
This warning indicated that the amount of shared memory allocated for copying data between processes is insufficient.
You need to increase the amount of allocated memory in the configuration file by setting `max_rowsize` under
`train.loader` or `eval.loader` (depending on your pipeline needs) to a larger value (default value is 64MB).

2. **I want to debug the data transformation pipeline, but the debugger doesn't stop at breakpoints. How can I debug?**</br>
To debug the data transformation pipeline, you need to set it to the debug mode first. You can do it by uncommenting
the `ms.dataset.config.set_debug_mode(True)` command located in
[mindocr/data/builder.py](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/builder.py). This will allow
the data pipeline to run synchronously and sequentially with a single thread.
