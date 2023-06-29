## ONNX 模型优化

### 1. 背景介绍

文本识别模型（例如CRNN）的最后一层为softmax，之后在后处理中使用numpy的argmax函数得出待识别字符在字典中的索引。

一方面对于识别结果来讲，不需要softmax计算出的概率，因此考虑将模型中的softmax节点删除,来加速推理过程。

另一方面，后处理逻辑中使用的numpy的argmax函数相较于模型中使用argmax算子耗时较大，因此考虑在ONNX模型中插入argmax算子来替代后处理中numpy的argmax函数。

### 2. 优化方法

优化前的ONNX模型结构如下所示：

<p align="center">
  <img src="https://user-images.githubusercontent.com/122354463/250898682-3a15ec8b-9c96-4582-877e-e843ea3dcffd.png" width=480 />
</p>
<p align="center">
  <em>优化前</em>
</p>

运行脚本 [insert_argmax.py](../../../deploy/models_utils/onnx_optim/insert_argmax.py), 传入参数model_path,为待优化的ONNX模型文件路径。

```python
python deploy/models_utils/onnx_optim/insert_argmax.py --model_path {path_to_model}
```
删除softmax节点，插入argmax节点，得到优化后的模型结构如下所示：

<p align="center">
  <img src="https://user-images.githubusercontent.com/122354463/250901029-c95a3120-b36b-486c-8952-97dcb3ab0ec8.png" width=480 />
</p>
<p align="center">
  <em>优化后</em>
</p>

经过测试，当输入batch_size 为16时，以ch_ppocr_server_v2.0_rec_infer.onnx模型为例，使用Ascend 310处理器，推理耗时降低了64.4%。
