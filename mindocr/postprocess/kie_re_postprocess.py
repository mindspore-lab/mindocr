import mindspore as ms

__all__ = ["VQAReTokenLayoutLMPostProcess"]


class VQAReTokenLayoutLMPostProcess:
    """Convert between text-label and text-index"""

    def __init__(self, **kwargs):
        super(VQAReTokenLayoutLMPostProcess, self).__init__()

    def __call__(self, logits, **kwargs):
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, ms.Tensor):
            logits = logits.numpy()
        return logits
