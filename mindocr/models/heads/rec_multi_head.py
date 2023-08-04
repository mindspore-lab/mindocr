from typing import Optional, Tuple, Union

from mindspore import Tensor, nn

from .rec_attn_head import AttentionHead
from .rec_ctc_head import CTCHead

__all__ = ["CTCAttnMultiHead"]


class CTCAttnMultiHead(nn.Cell):
    """A Multi Head of CTC + Attention"""

    def __init__(
        self,
        in_channels: Tuple[int, int],
        out_channels: int,
        attn_in_channels: int = 256,
        hidden_size: int = 256,
        batch_max_length: int = 25,
    ) -> None:
        super().__init__()
        self.ctc_head = CTCHead(in_channels[1], out_channels - 1)
        self.attn_head = AttentionHead(in_channels[0], out_channels, hidden_size, batch_max_length=batch_max_length)

    def construct(
        self, inputs: Tuple[Tensor, Tensor], attn_targets: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        attn_inputs, ctc_inputs = inputs

        if attn_targets is not None:
            # training branch
            ctc_output = self.ctc_head(ctc_inputs)
            attn_output = self.attn_head(attn_inputs, attn_targets)
            return ctc_output, attn_output
        else:
            # inference branch
            ctc_output = self.ctc_head(ctc_inputs)
            return ctc_output
