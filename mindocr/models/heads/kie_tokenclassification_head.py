from mindspore import nn, ops
from mindspore.common import float16, float32

from ..backbones.layoutxlm.configuration import LayoutXLMPretrainedConfig


class TokenClassificationHead(nn.Cell):
    def __init__(
        self,
        num_classes: int = 7,
        use_visual_backbone: bool = True,
        use_float16: bool = False,
        dropout_prod=None,
        **kwargs
    ):
        super(TokenClassificationHead, self).__init__()
        self.num_classes = num_classes
        dense_type = float32
        if use_float16 is True:
            dense_type = float16
        self.config = LayoutXLMPretrainedConfig(use_visual_backbone, use_float16)
        dropout_prod = dropout_prod if dropout_prod is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=dropout_prod)
        self.classifier = nn.Dense(self.config.hidden_size, num_classes).to_float(dense_type)

    @staticmethod
    def _init_weights(layer):
        """Initialize the weights"""
        if isinstance(layer, nn.Dense):
            layer.weight.set_data(ops.normal(shape=layer.weight.shape, mean=0.0, stddev=0.02))
            if layer.bias is not None:
                layer.bias.set_data(ops.zeros(size=layer.bias.shape))

    def construct(self, x, input_id=None):
        # sequence out and image out
        seq_length = input_id.shape[1]
        sequence_output = x[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits
