## 后处理模块指南

### 通用协议

1. 每个后处理模块是一个**class**，具有可调用的函数。
2. 后处理功能的输入是网络预测和附加数据信息（如果需要）。
3. 后处理功能的输出为总是一个dict，其中键为字段名，如文本检测中多边形为 'polys'，文本检测为'text'。

### 检测后处理API协议
1. class命名：Det{Method}Postprocess
2. class  `__init__()` args:
    - `box_type` (string):对于四边形和多边形文本表示，选项为["quad"，"polys"]。
    - `rescale_fields` (List[str]='polys'): 指示输出dict中的哪些字段将被重新缩放到原始图像空间。字段名称："polys"

3.  `__call__()`方法：如果继承自`DetBasePostprocess`，则不需要在Postproc class中实现此方法。
    后处理的执行项，对变换后的图像空间进行网络预测后处理，以获取文本框（通过`self._postprocess()`函数），然后将其重新缩放回原始图像空间（通过`self.rescale()`函数）。

    - 输入参数：
        - `pred` (Union[Tensor, Tuple[Tensor]]): 输入批次数据的网络预测，shape [batch_size, ...]
        - `shape_list` (Union[List, np.ndarray, ms.Tensor]): 批处理中每个图像的形状和比例信息，shape [batch_size, 4]。每个长度为4的内部数组是[src_h, src_w, scale_h, scale_w]，其中src_h和src_w是原始图像的高度和宽度，scale_h和scale_w分别是图像大小调整后的比例。
        - `**kwargs`: 扩展的参数

    - Return：检测结果作为字典，包含以下键
        - `polys` (List[List[np.ndarray]): 在**original**图像空间上映射的预测多边形，shape [batch_size, num_polygons, num_points, 2]。如果`box_type`为'quad'，num_points=4，则内部np.ndarray的shape [4, 2]
        - `scores` (List[float]): 预测多边形、shape (batch_size, num_polygons)的置信度得分

4. `_postprocess()`方法：如果继承自`DetBasePostprocess`，请在此处实现后处理方法
    后处理网络预测以获取转换后图像空间上的文本框（将在__call__函数中重新缩放回原始图像空间）

    - 输入参数：
        - `pred` (Union[Tensor, Tuple[Tensor]]): 输入批次数据的网络预测，shape [batch_size, ...]
        - `**kwargs`: 扩展的参数

    - Return：带键字典的后处理结果：
        - `polys` (List[List[np.ndarray]): 在**transformed**（即正常调整大小）图像空间上的预测多边形，shape (batch_size, num_polygons, num_points, 2)。如果`box_type`为"quad"，则num_points=4。
        - `scores` (List[float]): 预测多边形、shape (batch_size, num_polygons)的置信度得分

    - 注意事项：
        - 请将`pred`强制转换为实现中所需的类型。有些后处理步骤使用mindspore.nn中的ops，并且更适配张量类型，而有些步骤更适配其他库中所需的np.ndarray类型。
        - `_postprocess()` **NOT round**返回文本框`polys`为整数，因为它们将被重新设置，然后在最后取整。提前舍入将导致多边形重缩放中的较大错误，并导致**评估性能下降**，尤其是在小数据集上。

5. 关于将多边形重缩放回原始图像空间
    - 重缩放步骤对于公平评估是必要的，并且在推理中从原始图像裁剪文本区域时也是必要的。
    - 启用重新缩放以进行评估
        1. 在模型YAML配置文件的`eval.dataset.output_columns`中增加"shape_list"。
        2. 确保`rescale_fields`不是None（默认值为[“polys”]）
    - 要在推理中启用重缩放，请执行以下操作：
        1. 直接解析`shape_list`（数据加载后从数据["shape_list"]获取）到后处理函数。
            它与`rescale_fields`一起工作，以决定是否进行重缩放以及要重缩放哪些字段。
    - `shape_list`最初记录在图像调整大小转换中，例如`DetResize`。

**示例代码：** [DetBasePostprocess](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/det_base_postprocess.py)和[DetDBPostprocess](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/det_db_postprocess.py)

### 识别后处理API协议
1. class  `__init__()` should support the follow args:
        - character_dict_path
        - use_space_char
        - blank_at_last
        - lower
请参考API文档[RecCTCLabelDecode](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/rec_postprocess.py)用于参数说明。

2. `__call__()` method:
    - Input args:
        - `pred` (Union[Tensor, Tuple[Tensor]]): network prediction
        - `**kwargs`: args for extension

    - Return: det_res as a dictionary with the following keys
        - `texts` (List[str]): list of preditected text string
        - `confs` (List[float]): confidence of each prediction

**示例代码：**[RecCTCLabelDecode](https://github.com/mindspore-lab/mindocr/blob/main/mindocr/postprocess/rec_postprocess.py)
