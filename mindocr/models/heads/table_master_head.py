"""
This code is refer from:
https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/mmocr/models/textrecog/decoders/master_decoder.py
"""
import copy
import math

import mindspore as ms
from mindspore import Tensor, nn, ops


class TableMasterHead(nn.Cell):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """

    def __init__(self,
                 in_channels,
                 out_channels=43,
                 headers=8,
                 d_ff=2048,
                 dropout=0.,
                 max_text_length=500,
                 loc_reg_num=4,
                 share_parameter=False,
                 stacks=2,
                 **kwargs):
        super(TableMasterHead, self).__init__()
        hidden_size = in_channels
        self.layers = clones(DecoderLayer(headers, hidden_size, dropout, d_ff), 1 if share_parameter else stacks)
        self.cls_layer = DecoderLayer(headers, hidden_size, dropout, d_ff)
        self.bbox_layer = DecoderLayer(headers, hidden_size, dropout, d_ff)
        self.cls_fc = nn.Dense(hidden_size, out_channels)
        self.bbox_fc = nn.SequentialCell(
            nn.Dense(hidden_size, loc_reg_num),
            nn.Sigmoid())

        self.stacks = stacks
        self.SOS = out_channels - 3
        self.PAD = out_channels - 1
        self.loc_reg_num = loc_reg_num
        self.out_channels = out_channels
        self.max_text_length = max_text_length
        self.share_parameter = share_parameter

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm([hidden_size])
        self.embedding = Embeddings(hidden_size, out_channels)
        self.position = PositionalEncoding(hidden_size, dropout)

        self.tril = ops.tril
        self.argmax = ops.Argmax(axis=2)

    def make_mask(self, targets):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        target_pad_mask = targets != self.PAD
        target_pad_mask = target_pad_mask[:, None, :, None]
        target_pad_mask = ops.cast(target_pad_mask, ms.int32)
        target_length = targets.shape[1]
        target_sub_mask = self.tril(ops.ones((target_length, target_length), ms.int32))
        target_mask = ops.bitwise_and(target_pad_mask, target_sub_mask)
        return target_mask

    def decode(self, feature, targets, src_mask=None, tgt_mask=None):
        # main process of transformer decoder.
        targets = self.embedding(targets)
        targets = self.position(targets)
        output = targets
        for i in range(self.stacks):
            if self.share_parameter:
                actual_i = 0
            else:
                actual_i = i
            output = self.layers[actual_i](output, feature, src_mask, tgt_mask)

        # cls head
        cls_x = self.cls_layer(output, feature, src_mask, tgt_mask)
        cls_x = self.norm(cls_x)

        # bbox head
        bbox_x = self.bbox_layer(output, feature, src_mask, tgt_mask)
        bbox_x = self.norm(bbox_x)
        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def construct(self, feat, targets=None):
        N = feat.shape[0]
        num_steps = self.max_text_length + 1
        b, c, h, w = feat.shape
        feat = feat.reshape([b, c, h * w])  # flatten 2D feature map
        feat = feat.transpose((0, 2, 1))
        out_enc = self.position(feat)
        if targets is not None:
            # training branch
            targets = targets[0]
            targets = targets[:, :-1]
            target_mask = self.make_mask(targets)
            output, bbox_output = self.decode(out_enc, targets, tgt_mask=target_mask)
            return output, bbox_output
        else:
            input = ops.zeros((N, 1), ms.int32) + self.SOS
            output = ops.zeros(
                [input.shape[0], self.max_text_length + 1, self.out_channels])
            bbox_output = ops.zeros(
                [input.shape[0], self.max_text_length + 1, self.loc_reg_num])
            # probs = list()
            for i in range(num_steps):
                target_mask = self.make_mask(input)
                out_step, bbox_output_step = self.decode(out_enc, input, tgt_mask=target_mask)
                prob = ops.softmax(out_step, axis=-1)
                next_word = self.argmax(prob)
                input = ops.concat(
                    [input, next_word[:, -1].unsqueeze(-1)], axis=1)
                # probs.append(probs_step[:, i])
                if i == self.max_text_length:
                    output = out_step
                    bbox_output = bbox_output_step
            # probs = ops.stack(probs, axis=1)
            output = ops.softmax(output, axis=-1)
        return output, bbox_output


class LayerNormLayer(nn.Cell):
    def __init__(self, dims):
        super(LayerNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm[dims]

    def construct(self, x):
        return self.layer_norm(x)


class DecoderLayer(nn.Cell):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """

    def __init__(self, headers, d_model, dropout, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(headers, d_model, dropout)
        self.src_attn = MultiHeadAttention(headers, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SubLayerConnection(d_model, dropout), 3)

    def construct(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Cell):
    def __init__(self, headers, d_model, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Dense(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.batch_matmul = ops.BatchMatMul()

    def construct(self, query, key, value, mask=None):
        B = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [layer(x).reshape([B, -1, self.headers, self.d_k]).transpose([0, 2, 1, 3])
             for layer, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, attn = self.self_attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose([0, 2, 1, 3]).reshape([B, -1, self.headers * self.d_k])
        return self.linears[-1](x)

    def self_attention(self, query, key, value, mask=None, dropout=None):
        """
        Compute 'Scale Dot Product Attention'
        """
        d_k = float(value.shape[-1])
        d_k_sqrt = ops.cast(ms.numpy.sqrt(d_k), query.dtype)
        score = self.batch_matmul(query, key.transpose([0, 1, 3, 2])) / d_k_sqrt
        if mask is not None:
            # score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
            score = ops.masked_fill(score.astype(ms.float32), mask == 0, -6.55e4)

        p_attn = ops.softmax(score, axis=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return ops.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Cell):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Dense(d_model, d_ff)
        self.w_2 = nn.Dense(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x):
        return self.w_2(self.dropout(ops.relu(self.w_1(x))))


class SubLayerConnection(nn.Cell):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm([size])
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """ Produce N identical layers """
    return nn.CellList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Cell):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.sqrt_d_model = ops.sqrt(Tensor(d_model, ms.float32))

    def construct(self, *input):
        x = input[0]
        return self.lut(x) * self.sqrt_d_model


class PositionalEncoding(nn.Cell):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = ops.zeros([max_len, d_model])
        position = ops.arange(0, max_len).unsqueeze(1).astype('float32')
        div_term = ops.exp(
            ops.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def construct(self, feat, **kwargs):
        feat = feat + self.pe[:, :ops.shape(feat)[1]]  # pe 1*5000*512
        return self.dropout(feat)
