import mindspore as ms
from mindspore import Tensor


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
