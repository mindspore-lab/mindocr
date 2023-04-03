"""Cosine Decay with Warmup Learning Rate Scheduler"""
from mindspore import nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class WarmupCosineDecayLR(LearningRateSchedule):
    """CosineDecayLR with warmup
    Args:

        min_lr: (float) lower lr bound for 'WarmupCosineDecayLR' schedulers.
        max_lr: (float) upper lr bound for 'WarmupCosineDecayLR' schedulers.
        warmup_epochs: (int) the number of warm up epochs of learning rate.
        decay_epochs: (int) the number of decay epochs of learning rate.
        steps_per_epoch: (int) the number of steps per epoch.
        step_mode: (bool) determine decay along steps or epochs. True for steps, False for epochs.

    The learning rate will increase from 0 to max_lr in `warmup_epochs` epochs,
    then decay to min_lr in `decay_epochs` epochs
    """

    def __init__(
        self,
        min_lr,
        max_lr,
        warmup_epochs,
        decay_epochs,
        steps_per_epoch,
        step_mode=True,
    ):
        super().__init__()
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = decay_epochs * steps_per_epoch
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.step_mode = step_mode
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(max_lr, self.warmup_steps if step_mode else warmup_epochs)
        self.cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, self.decay_steps if step_mode else decay_epochs)

    def step_lr(self, global_step):
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(global_step - self.warmup_steps)
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.cosine_decay_lr(global_step)
        return lr

    def epoch_lr(self, global_step):
        cur_epoch = global_step // self.steps_per_epoch
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(cur_epoch - self.warmup_epochs)
            else:
                lr = self.warmup_lr(cur_epoch)
        else:
            lr = self.cosine_decay_lr(cur_epoch)
        return lr

    def construct(self, global_step):
        if self.step_mode:
            lr = self.step_lr(global_step)
        else:
            lr = self.epoch_lr(global_step)

        return lr
