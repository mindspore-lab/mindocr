from packaging import version

import mindspore as ms
from mindspore import Tensor
from mindspore.ops.primitive import constexpr


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = Tensor(0.0, dtype=ms.float32)
        self.avg = Tensor(0.0, dtype=ms.float32)
        self.sum = Tensor(0.0, dtype=ms.float32)
        self.count = Tensor(0.0, dtype=ms.float32)

    def update(self, val: Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fetch_optimizer_lr(opt):
    # print(f"Before, global step: {opt.global_step}")
    lr = opt.learning_rate
    if opt.dynamic_lr:
        if opt.is_group_lr:
            lr = ()
            for learning_rate in opt.learning_rate:
                cur_dynamic_lr = learning_rate(opt.global_step - 1).reshape(())
                lr += (cur_dynamic_lr,)
        else:
            lr = opt.learning_rate(opt.global_step - 1).reshape(())
    # print(f"After, global step: {opt.global_step}")
    return lr


@constexpr
def is_ms_version_2():
    """This check can be applied in `nn.Cell.construct` method, to
    make compatibilities in differenct Mindspore version
    """
    return version.parse(ms.__version__) >= version.parse("2.0.0rc")
