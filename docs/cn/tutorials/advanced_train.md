# 进阶训练策略

### 策略：梯度累积，梯度裁剪，EMA

训练策略可在模型YAML配置文件中进行配置。请在设置后运行`tools/train.py`脚本进行训练

[Yaml配置文件参考样例](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_icdar15.yaml)

```yaml
train:
  gradient_accumulation_steps: 2
  clip_grad: True
  clip_norm: 5.0
  ema: True
  ema_decay: 0.9999
```

#### 梯度累积

梯度累积可以有效解决显存不足的问题，使得在同等显存，允许**使用更大的全局batch size进行训练**。可以通过在yaml配置中将`train.gradient_accumulation_steps` 设置为大于1的值来启用梯度累积功能。
等价的全局batch size为：


`global_batch_size = batch_size * num_devices * gradient_accumulation_steps`

#### 梯度裁剪

梯度裁剪通常用来缓解梯度爆炸/溢出问题，以使模型收敛更稳定。可以通过在yaml配置中设置`train.clip_grad`为`True`来启用该功能，调整`train.clip_norm`的值可以控制梯度裁剪范数的大小。


#### EMA

Exponential Moving Average（EMA）是一种平滑模型权重的模型集成方法。它能帮助模型在训练中稳定收敛，并且通常会带来更好的模型性能。
可以通过在yaml配置中设置`train.ema`为`True`来使用该功能，并且可以调整`train.ema_decay`来控制权重衰减率，通常设置为接近1的值.


### 断点续训

断点续训通常用于训练意外中断时，此时使用该功能可以继续从中断处epoch继续训练。可以通过在yaml配置中设置`model.resume`为`True`来使用该功能，用例如下：

```yaml
model:
  resume: True
```
>
>默认情况下，它将从`train.ckpt_save_dir`目录中保存的`train_resume.ckpt`恢复。

如果要使用其他epoch用于恢复训练，请在`resume`中指定epoch路径，用例如下：

```yaml
model:
  resume: /some/path/to/train_resume.ckpt
```

### OpenI云平台训练

请参考[MindOCR云上训练快速入门](training_on_openi.md)
