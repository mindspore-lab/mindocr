from mindspore import nn, ops
from mindspore.common.dtype import float16, float32

from ..backbones.layoutxlm.configuration import LayoutXLMPretrainedConfig


class BiaffineAttention(nn.Cell):
    """Implements a biaffine attention operator for binary relation classification."""

    def __init__(self, in_features, out_features, use_float16: bool = True):
        super(BiaffineAttention, self).__init__()
        self.dense_dtype = float16 if use_float16 else float32
        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = nn.BiDense(in_features, in_features, out_features, has_bias=False)
        self.linear = nn.Dense(2 * in_features, out_features).to_float(self.dense_dtype)

    def construct(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(ops.concat((x_1, x_2), axis=-1))


class REDecoder(nn.Cell):
    """
    Decoder of relation extraction
    """
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1, use_float16: bool = True):
        super(REDecoder, self).__init__()
        self.dense_dtype = float16 if use_float16 else float32
        self.entity_emb = nn.Embedding(3, hidden_size)
        self.ffnn_head = nn.SequentialCell(
            nn.Dense(hidden_size * 2, hidden_size).to_float(self.dense_dtype),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
            nn.Dense(hidden_size, hidden_size // 2).to_float(self.dense_dtype),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
        )
        self.ffnn_tail = nn.SequentialCell(
            nn.Dense(hidden_size * 2, hidden_size).to_float(self.dense_dtype),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
            nn.Dense(hidden_size, hidden_size // 2).to_float(self.dense_dtype),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
        )
        self.rel_classifier = BiaffineAttention(hidden_size // 2, 2)

    def construct(self, hidden_states, question, question_label, answer, answer_label):
        __, _, hidden_size = hidden_states.shape
        q_label_repr = self.entity_emb(question_label)
        question = question.expand_dims(-1).repeat_interleave(hidden_size, -1)
        tmp_hidden_states = ops.gather_d(hidden_states, 1, question)
        q_repr = ops.concat((tmp_hidden_states, q_label_repr), axis=-1)

        a_label_repr = self.entity_emb(answer_label)
        answer = answer.expand_dims(-1).repeat_interleave(hidden_size, -1)
        tmp_hidden_states = ops.gather_d(hidden_states, 1, answer)
        a_repr = ops.concat((tmp_hidden_states, a_label_repr), axis=-1)

        q = self.ffnn_head(q_repr).astype(float32)
        a = self.ffnn_tail(a_repr).astype(float32)
        logits = self.rel_classifier(q, a)
        return logits


class RelationExtractionHead(nn.Cell):
    """
    Head of relation extraction tas
    """
    def __init__(self, use_visual_backbone: bool = True, use_float16: bool = False, dropout=None, **kwargs):
        super(RelationExtractionHead, self).__init__()
        self.config = LayoutXLMPretrainedConfig(use_visual_backbone, use_float16)

        dropout_prob = dropout if dropout is not None else self.config.hidden_dropout_prob

        self.extractor = REDecoder(self.config.hidden_size, dropout_prob, use_float16)

        self.dropout = nn.Dropout(p=dropout_prob)

    def construct(
        self,
        backbone_out,
        input_ids,
        question=None,
        question_label=None,
        answer=None,
        answer_label=None,
    ):
        seq_length = input_ids.shape[1]
        sequence_output = backbone_out[0][:, :seq_length]

        sequence_output = self.dropout(sequence_output)
        logits = self.extractor(sequence_output, question, question_label, answer, answer_label)

        return logits
