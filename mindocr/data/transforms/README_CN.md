## 定制化数据转换开发指导

### 写法指导

1. 每个转换都是一个具有可调用函数的类。示例如下所示。

2. 转换函数的输入总是一个字典，包含img_path, raw label等数据信息。

3. 请为__call__函数编写注释，以澄清数据字典中required/modified/added中所需键。

4. 在类init函数中添加kwargs进行扩展，用于解析全局配置，如is_train。

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

### 添加单元测试和可视化

请在`tests/ut/transforms`中为转换添加单元测试，并尝试覆盖不同的情况（输入和设置）。

请使用jupyter notebook 可视的检查图像和标注转换的正确性。请参见`transform_tutorial.ipynb`。

### 重要注意事项
1. 对于文本检测推理或评估中使用的空间转换操作（如确定性调整大小、缩放），请将空间转换信息记录在`shape_list`中。否则，后处理方法将无法将结果映射回原始图像空间。关于如何记录`shape_list`，请参阅[DetResize](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/data/transforms/det_transforms.py)。
