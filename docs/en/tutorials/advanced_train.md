# Advanced Training

### Tricks: Gradient Accumulation, Gradient Clipping, and EMA

All the training skills can be configured in the model config file.
After setting, please run `tools/train.py` script to trigger training.

[Example Yaml Config](configs/rec/crnn/crnn_icdar15.yaml)

```yaml
train:
  gradient_accumulation_steps: 2
  clip_grad: True
  clip_norm: 5.0
  ema: True
  ema_decay: 0.9999
```

#### Gradient Accumulation

Gradient accumulation is an effective skill to address  memory limitation issue and allow **training with large global batch size**.

To enable it, please set `train.gradient_accumulation_steps` with value larger than 1 in yaml config.

The equivalent global batch size would be
`global_batch_size = batch_size * num_devices * gradient_accumulation_steps`

#### Gradient Clipping

Gradient clipping is a method to address gradient explosion/overflow problem and to
stabilize model convergence.

To enable it, please set `train.ema` as True and optionally adjust the norm value by `train.clip_norm`.


#### EMA

Exponential Moving Average (EMA) can be viewed as a model ensemble method that smooth the model weights.
It can help stabilize model convergence in training and usually leads to better model performance.

To enable it, please set `train.ema` as True. You may also adjust `train.ema_decay` to control the decay rate.

### Resume Training

Resume training is useful when the training is interrupted unexpectedly.

To resume training, please set `model.resume` as True in the yaml config as follows.
```yaml
mode:
  resume: True
```
> By default, it will resume from the "train_resume.ckpt" checkpoint file in the directory defined by `train.ckpt_save_dir` .


If you want to specify another checkpoint to resume from, please parse the checkpoint path to `resume`, for example

```yaml
mode:
  resume: /some/path/to/train_resume.ckpt
```

### Training on OpenI Cloud Platform

Please refer to the [MindOCR OpenI Training Guideline](../../cn/tutorials/training_on_openi.md)
