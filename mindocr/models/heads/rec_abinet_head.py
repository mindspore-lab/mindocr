import mindspore as ms
from mindspore import nn

from ..utils.abinet_layers import ABINetBlock

__all__ = ["ABINetHead"]


class ABINetHead(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.iter_size = 3
        self.in_channels = in_channels  # In order to fit the mindocr framework, it is not actually used.
        self.alignment = BaseAlignment()

    def construct(self, nout):
        all_l_res, v_res = nout[0], nout[1]
        a_res = v_res
        all_a_res = []
        for i in range(self.iter_size):
            l_res = all_l_res[i]
            a_res = self.alignment(l_res["feature"], v_res["feature"])
            all_a_res.append(a_res)

        return all_a_res, all_l_res, v_res


class BaseAlignment(ABINetBlock):
    def __init__(self):
        super().__init__()
        d_model = 512

        self.loss_weight = 1.0
        self.max_length = 26  # additional stop token
        self.w_att = nn.Dense(
            2 * d_model, d_model, weight_init="uniform", bias_init="uniform"
        )
        self.cls = nn.Dense(
            d_model,
            self.charset.num_classes,
            weight_init="uniform",
            bias_init="uniform",
        )

    def construct(self, l_feature, v_feature):

        f = ms.ops.concat((l_feature, v_feature), axis=2)
        f_att = ms.ops.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {
            "logits": logits,
            "pt_lengths": pt_lengths,
            "loss_weight": self.loss_weight,
            "name": "alignment",
        }
