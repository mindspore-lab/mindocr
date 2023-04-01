"""MultiStep Decay Learning Rate Scheduler"""
import mindspore as ms
from mindspore import nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class MultiStepDecayLR(LearningRateSchedule):
    """Multiple step learning rate
    The learning rate will decay once the number of step reaches one of the milestones.
    """

    def __init__(self, lr, warmup_epochs, decay_rate, milestones, steps_per_epoch, num_epochs):
        super().__init__()
        self.warmup_steps = warmup_epochs * steps_per_epoch
        num_steps = num_epochs * steps_per_epoch
        step_lrs = []
        cur_lr = lr
        k = 0
        for step in range(num_steps):
            if step == milestones[k] * steps_per_epoch:
                cur_lr = cur_lr * decay_rate
                k = min(k + 1, len(milestones) - 1)
            step_lrs.append(cur_lr)
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(lr, self.warmup_steps)
        self.step_lrs = ms.Tensor(step_lrs, ms.float32)

    def construct(self, global_step):
        if self.warmup_steps > 0 and global_step < self.warmup_steps:
            lr = self.warmup_lr(global_step)
        elif global_step < self.step_lrs.shape[0]:
            lr = self.step_lrs[global_step]
        else:
            lr = self.step_lrs[-1]
        return lr
