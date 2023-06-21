import mindspore.nn as nn

from ...models.utils.abinet_utils import ABINetBlock, PositionAttention, ResTranformer
from ._registry import register_backbone, register_backbone_class

__all__ = [
    "ABINetIterBackbone",
    "abinet_backbone"]


# ABINet_backbone


@register_backbone_class
class ABINetIterBackbone(nn.Cell):
    def __init__(self, batchsize=96):
        super().__init__()
        self.out_channels = 512
        self.batchsize = batchsize
        self.vision = BaseVision(self.batchsize)

    def construct(self, images, *args):
        v_res = self.vision(images)
        return v_res


@register_backbone
def abinet_backbone(pretrained: bool = True, **kwargs):
    model = ABINetIterBackbone(**kwargs)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `rec_abinet_backbone` backbone does not exist")

    return model


class BaseVision(ABINetBlock):
    def __init__(self, batchsize):
        super().__init__()
        self.batchsize = batchsize
        self.loss_weight = 1.0
        self.out_channels = 512
        self.backbone = ResTranformer(self.batchsize)
        mode = "nearest"
        self.attention = PositionAttention(
            max_length=26,  # additional stop token
            mode=mode,
        )

        self.cls = nn.Dense(
            self.out_channels,
            self.charset.num_classes,
            weight_init="uniform",
            bias_init="uniform",
        )

    def construct(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)

        attn_vecs, attn_scores = self.attention(features)

        logits = self.cls(attn_vecs)  # (N, T, C)

        pt_lengths = self._get_length(logits)

        return {
            "feature": attn_vecs,
            "logits": logits,
            "pt_lengths": pt_lengths,
            "attn_scores": attn_scores,
            "loss_weight": self.loss_weight,
            "name": "vision",
        }
