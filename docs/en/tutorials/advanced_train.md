# Advanced Training

### Tricks: Gradient Accumulation, Gradient Clipping, and EMA

All the training tricks can be configured in the model config files.
After setting, please run `tools/train.py` script to initiate training.

[Example Yaml Config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_icdar15.yaml)

```yaml
train:
  gradient_accumulation_steps: 2
  clip_grad: True
  clip_norm: 5.0
  ema: True
  ema_decay: 0.9999
```

#### Gradient Accumulation

Gradient accumulation is an effective way to address  memory limitation issue and allow **training with large global batch size**.

To enable it, set `train.gradient_accumulation_steps` to values larger than 1 in yaml config.

The equivalent global batch size would be
`global_batch_size = batch_size * num_devices * gradient_accumulation_steps`

#### Gradient Clipping

Gradient clipping is a method to address gradient explosion/overflow problem and
stabilize model convergence.

To enable it, set `train.ema` to `True` and optionally adjust the norm value in `train.clip_norm`.


#### EMA

Exponential Moving Average (EMA) can be viewed as a model ensemble method that smooths the model weights.
It can help stabilize model convergence in training and usually leads to better model performance.

To enable it, set `train.ema` to `True`. You may also adjust `train.ema_decay` to control the decay rate.

### Resume Training

Resuming training is useful when the training was interrupted unexpectedly.

To resume training, set `model.resume` to `True` in the yaml config as follows:
```yaml
model:
  resume: True
```
> By default, it will resume from the "train_resume.ckpt" checkpoint file located in the directory
> specified in `train.ckpt_save_dir`.


If you want to use another checkpoint to resume from, specify the checkpoint path in `resume` as follows:

```yaml
model:
  resume: /some/path/to/train_resume.ckpt
```

### Training on OpenI Cloud Platform

Please refer to the [MindOCR OpenI Training Guideline](training_on_openi.md)
