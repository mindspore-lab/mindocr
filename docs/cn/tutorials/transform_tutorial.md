# Transformation教程

[![Download Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindocr/tutorials/transform_tutorial.ipynb)&emsp;

## 机制

1. MindOCR支持两种类型的Transformation：原生MindSpore（如`Decode`、`HWC2CHW`等）和自定义python转换。MindOCR提供了许多python转换，适用于OCR中的各种应用。

2. Python转换类必须有一个`__call__` 方法，该方法的输入和输出都是字典，如下例所示：


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
3. 输入/输出字典包含诸如`image`、`polys`等数据键。transformation可以修改数据字典或向数据字典添加新值，但不应从中删除键（数据）。
4. 每个transformation类都必须有一个`self.output_columns`，这对于数据映射流程是必要的。
5. ⚠: 为了提高transformation pipeline的效率，python和MindSpore转换在配置文件中应该各自组合在一起，这样可以最大限度地减少它们组成的组的数量。
也就是说，python操作之后应该是python操作，MindSpore操作之后是MindSpor操作。例如：

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
在上面的例子中，transformation组的最小可能数量是3（在对图像应用transformation之前，我们必须首先`Decode`图像）。

可用的transformation可在`mindocr/data/transforms/*_transform.py`中找到。

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

## 文本检测

### 1. 加载图像和注释

#### 准备


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


#### Mindspore transform: Decode


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


### 2. 图像和注释处理/增强

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


## 常见问题
1. **在程序执行过程中，我看到以下警告: `Using shared memory queue, but rowsize is larger than allocated
memory max_rowsize: X MB, current rowsize: X MB`. 我该怎么解决这个问题?**</br>
此警告是因为在进程之间复制数据而分配的共享内存量不足。您需要通过将`train.loader`或`eval.loader`下（取决于您的流程需求）的`max_rowsize`设置为更大的值（默认值为64MB）来增加配置文件中分配的内存量。

2. **我想调试data transformation pipeline，但调试器不会在断点处停止。如何调试？**</br>
要调试data transformation pipeline，需要首先将其设置为调试模式。 您可以通过取消位于[mindocr/data/builder.py](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/builder.py)
中的`ms.dataset.config.set_debug_mode(True)`命令来完成此操作。这将允许data pipeline单线程同步并按顺序运行。
