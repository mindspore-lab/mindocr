import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import nn


class BaseDecoder(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def construct(self,
                  feat,
                  out_enc,
                  label=None,
                  valid_width_masks=None,
                  word_positions=None,
                  train_mode=True):
        # self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, valid_width_masks, word_positions)
        return self.forward_test(feat, out_enc, valid_width_masks, word_positions)


class ChannelReductionEncoder(nn.Cell):
    """Change the channel number with a one by one convoluational layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(ChannelReductionEncoder, self).__init__()

        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def construct(self, feat):
        """
        Args:
            feat (Tensor): Image features with the shape of
                :math:`(N, C_{in}, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        return self.layer(feat)


class DotProductAttentionLayer(nn.Cell):

    def __init__(self, dim_model=None):
        super().__init__()

        self.scale = dim_model ** -0.5 if dim_model is not None else 1.

        self.transpose = ops.Transpose()
        self.matmul = ops.MatMul()
        self.batchmatmul = ops.BatchMatMul()
        self.softmax = ops.Softmax(axis=2)

    def construct(self, query, key, value, h, w, valid_width_masks=None):
        query = self.transpose(query, (0, 2, 1))
        logits = self.batchmatmul(query, key) * self.scale

        n, c, t = logits.shape
        # reshape to (n, c, h, w)
        logits = logits.view((n, c, h, w))
        if valid_width_masks is not None:
            logits = ops.split(logits, 1, axis=0)
            logits_list = []
            for i, valid_width_mask in enumerate(valid_width_masks):
                logits_i = logits[i].squeeze(0)  # (c, h, w)
                logits_i = logits_i.view((-1, w))  # (c*h, w)
                ch = c * h
                valid_width_mask = valid_width_mask.repeat_interleave(ch, 0)  # (c*h, w)
                valid_width_mask = ops.cast(valid_width_mask, ms.bool_)
                logits_i = ops.select(valid_width_mask, logits_i, float('-inf'))  # (c*h, w)
                logits_list.append(logits_i.view((c, h, w)))  # (c, h, w)
            logits = ops.concat(logits_list, axis=0)
        # reshape to (n, c, h, w)
        logits = logits.view((n, c, t))

        weights = self.softmax(logits)

        value = self.transpose(value, (0, 2, 1))
        glimpse = self.batchmatmul(weights, value)
        glimpse = self.transpose(glimpse, (0, 2, 1))
        return glimpse


class SequenceAttentionDecoder(BaseDecoder):
    """Sequence attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        rnn_layers (int): Number of RNN layers.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_width_mask']``.
        padding_idx (int): The index of `<PAD>`.
        dropout (float): Dropout rate.
        return_feature (bool): Return feature or logits as the result.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    """

    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 start_idx=0,
                 mask=True,
                 padding_idx=None,
                 dropout=0.,
                 return_feature=False,
                 encode_value=False):
        super().__init__()

        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.max_seq_len = max_seq_len
        self.start_idx = start_idx
        self.mask = mask

        self.transpose = ops.Transpose()
        self.ones = ops.Ones()
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.stack = ops.Stack()
        self.softmax = ops.Softmax(axis=-1)

        self.embedding = nn.Embedding(
            self.num_classes, self.dim_model, padding_idx=padding_idx)

        self.sequence_layer = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            dropout=dropout)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Dense(
                dim_model if encode_value else dim_input, pred_num_classes)

    def forward_train(self, feat, out_enc, targets, valid_width_masks):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets (Tensor): a tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_width_masks (Tensor): valid length ratio of img.
        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """

        tgt_embedding = self.embedding(targets)

        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        _, len_q, c_q = tgt_embedding.shape
        assert c_q == self.dim_model
        assert len_q <= self.max_seq_len

        query, _ = self.sequence_layer(tgt_embedding)
        query = self.transpose(query, (0, 2, 1))
        key = out_enc.view((n, c_enc, h * w))
        if self.encode_value:
            value = key
        else:
            value = feat.view((n, c_feat, h * w))

        attn_out = self.attention_layer(query, key, value, h, w, valid_width_masks)
        attn_out = self.transpose(attn_out, (0, 2, 1))

        if self.return_feature:
            return attn_out

        out = self.prediction(attn_out)

        return out

    def forward_test(self, feat, out_enc, valid_width_masks):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_width_masks (Tensor): valid length ratio of img.

        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C-1)`.
        """
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]

        decode_sequence = (self.ones((batch_size, seq_len), mstype.int64) * self.start_idx)

        outputs = []
        for i in range(seq_len):
            step_out = self.forward_test_step(feat, out_enc, decode_sequence,
                                              i, valid_width_masks)
            outputs.append(step_out)
            max_idx, _ = self.argmax(step_out)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        outputs = self.stack(outputs, 1)

        return outputs

    def forward_test_step(self, feat, out_enc, decode_sequence, current_step,
                          valid_width_masks):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            decode_sequence (Tensor): Shape :math:`(N, T)`. The tensor that
                stores history decoding result.
            current_step (int): Current decoding step.
            valid_width_masks (Tensor): valid length ratio of img

        Returns:
            Tensor: Shape :math:`(N, C-1)`. The logit tensor of predicted
            tokens at current time step.
        """

        embed = self.embedding(decode_sequence)

        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        _, _, c_q = embed.shape
        assert c_q == self.dim_model

        query, _ = self.sequence_layer(embed)
        query = self.transpose(query, (0, 2, 1))
        key = out_enc.view((n, c_enc, h * w))
        if self.encode_value:
            value = key
        else:
            value = feat.view((n, c_feat, h * w))

        attn_out = self.attention_layer(query, key, value, h, w, valid_width_masks)
        out = attn_out[:, :, current_step]

        if self.return_feature:
            return out

        out = self.prediction(out)
        # out = F.softmax(out, dim=-1)
        out = self.softmax(out)

        return out


class PositionAwareLayer(nn.Cell):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()

        self.dim_model = dim_model

        self.transpose = ops.Transpose()

        self.rnn = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers)

        self.mixer = nn.SequentialCell(
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1, pad_mode='pad'))

    def construct(self, img_feature):
        n, c, h, w = img_feature.shape

        rnn_input = self.transpose(img_feature, (0, 2, 3, 1))
        rnn_input = rnn_input.view((n * h, w, c))
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view((n, h, w, c))
        rnn_output = self.transpose(rnn_output, (0, 3, 1, 2))

        out = self.mixer(rnn_output)

        return out


class PositionAttentionDecoder(BaseDecoder):
    """Position attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        rnn_layers (int): Number of RNN layers.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_width_mask']``.
        return_feature (bool): Return feature or logits as the result.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss

    """

    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 mask=True,
                 return_feature=False,
                 encode_value=False):
        super().__init__()

        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask

        self.transpose = ops.Transpose()
        self.stack = ops.Stack()

        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)

        self.position_aware_module = PositionAwareLayer(
            self.dim_model, rnn_layers)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Dense(
                dim_model if encode_value else dim_input, pred_num_classes)

    def _get_position_index(self, length, batch_size):
        position_index_list = []
        for i in range(batch_size):
            position_index = np.arange(0, stop=length, step=1, dtype='int64')
            position_index_list.append(position_index)
        batch_position_index = self.stack(position_index_list)
        return batch_position_index

    def forward_train(self, feat, out_enc, targets, valid_width_masks, position_index):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_width_masks (Tensor): valid length ratio of img.
            position_index (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it will be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
        #
        n, c_enc, h, w = out_enc.shape  # size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape  # size()
        assert c_feat == self.dim_input
        _, len_q = targets.shape  # size()
        assert len_q <= self.max_seq_len

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = self.transpose(query, (0, 2, 1))
        key = position_out_enc.view((n, c_enc, h * w))
        if self.encode_value:
            value = out_enc.view((n, c_enc, h * w))
        else:
            value = feat.view((n, c_feat, h * w))

        attn_out = self.attention_layer(query, key, value, h, w, valid_width_masks)
        attn_out = self.transpose(attn_out, (0, 2, 1))  # [n, len_q, dim_v]

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)

    def forward_test(self, feat, out_enc, valid_width_masks, position_index):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_width_masks (Tensor): valid length ratio of img
            position_index (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = self.transpose(query, (0, 2, 1))
        key = position_out_enc.view((n, c_enc, h * w))
        if self.encode_value:
            value = out_enc.view((n, c_enc, h * w))
        else:
            value = feat.view((n, c_feat, h * w))

        attn_out = self.attention_layer(query, key, value, h, w, valid_width_masks)
        attn_out = self.transpose(attn_out, (0, 2, 1))  # [n, len_q, dim_v]

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)


class RobustScannerFusionLayer(nn.Cell):

    def __init__(self, dim_model, dim=-1):
        super(RobustScannerFusionLayer, self).__init__()

        self.dim_model = dim_model
        self.dim = dim
        self.linear_layer = nn.Dense(dim_model * 2, dim_model * 2)
        self.concat = ops.Concat(axis=self.dim)
        self.split = ops.Split(dim, 2)
        self.sigmoid = ops.Sigmoid()
        self.mul = ops.Mul()

    def construct(self, x0, x1):
        assert x0.shape == x1.shape
        fusion_input = self.concat([x0, x1])
        output = self.linear_layer(fusion_input)
        output_split = self.split(output)
        a1 = output_split[0]
        a2 = self.sigmoid(output_split[1])
        output = self.mul(a1, a2)
        return output


class RobustScannerDecoder(BaseDecoder):
    """Decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_width_mask']``.
        padding_idx (int): The index of `<PAD>`.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 num_classes=None,
                 dim_input=512,
                 dim_model=128,
                 hybrid_decoder_rnn_layers=2,
                 hybrid_decoder_dropout=0.,
                 position_decoder_rnn_layers=2,
                 max_seq_len=40,
                 start_idx=0,
                 mask=True,
                 padding_idx=None,
                 encode_value=False):
        super().__init__()
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.encode_value = encode_value
        self.start_idx = start_idx
        self.padding_idx = padding_idx
        self.mask = mask

        self.ones = ops.Ones()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.stack = ops.Stack(axis=1)

        # init hybrid decoder
        self.hybrid_decoder = SequenceAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=hybrid_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            dropout=hybrid_decoder_dropout,
            encode_value=encode_value,
            return_feature=True
        )

        # init position decoder
        self.position_decoder = PositionAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=position_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            mask=mask,
            encode_value=encode_value,
            return_feature=True
        )

        self.fusion_module = RobustScannerFusionLayer(
            self.dim_model if encode_value else dim_input)

        pred_num_classes = num_classes - 1
        self.prediction = nn.Dense(dim_model if encode_value else dim_input,
                                   pred_num_classes)

    def forward_train(self, feat, out_enc, target, valid_width_masks, word_positions):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            target (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_width_masks (Tensor):
            word_positions (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)`.
        """
        hybrid_glimpse = self.hybrid_decoder.forward_train(
            feat, out_enc, target, valid_width_masks)
        position_glimpse = self.position_decoder.forward_train(
            feat, out_enc, target, valid_width_masks, word_positions)

        fusion_out = self.fusion_module(hybrid_glimpse, position_glimpse)

        out = self.prediction(fusion_out)

        return out

    def forward_test(self, feat, out_enc, valid_width_masks, word_positions):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_width_masks (Tensor):
            word_positions (Tensor): The position of each word.
        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C-1)`.
        """
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]

        decode_sequence = (self.ones((batch_size, seq_len), mstype.int64) * self.start_idx)

        position_glimpse = self.position_decoder.forward_test(
            feat, out_enc, valid_width_masks, word_positions)

        outputs = []
        for i in range(seq_len):
            hybrid_glimpse_step = self.hybrid_decoder.forward_test_step(
                feat, out_enc, decode_sequence, i, valid_width_masks)

            fusion_out = self.fusion_module(hybrid_glimpse_step,
                                            position_glimpse[:, i, :])

            char_out = self.prediction(fusion_out)
            char_out = self.softmax(char_out)
            outputs.append(char_out)
            max_idx, _ = self.argmax(char_out)
            if i < seq_len - 1:
                decode_sequence = ops.cast(decode_sequence, ms.int8)
                max_idx = ops.cast(max_idx, ms.int8)
                decode_sequence[:, i + 1] = max_idx
                decode_sequence = ops.cast(decode_sequence, ms.int64)

        outputs = self.stack(outputs)

        return outputs


class RobustScannerHead(nn.Cell):
    def __init__(self,
                 out_channels,  # 90 + unknown + start + padding
                 in_channels,
                 enc_outchannles=128,
                 hybrid_dec_rnn_layers=2,
                 hybrid_dec_dropout=0.,
                 position_dec_rnn_layers=2,
                 start_idx=0,
                 max_seq_len=40,
                 mask=True,
                 padding_idx=None,
                 encode_value=False,
                 **kwargs):
        super(RobustScannerHead, self).__init__()

        # encoder module
        self.encoder = ChannelReductionEncoder(
            in_channels=in_channels, out_channels=enc_outchannles)

        # decoder module
        self.decoder = RobustScannerDecoder(
            num_classes=out_channels,
            dim_input=in_channels,
            dim_model=enc_outchannles,
            hybrid_decoder_rnn_layers=hybrid_dec_rnn_layers,
            hybrid_decoder_dropout=hybrid_dec_dropout,
            position_decoder_rnn_layers=position_dec_rnn_layers,
            max_seq_len=max_seq_len,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            encode_value=encode_value)

    def construct(self, inputs, targets):
        """
        targets: [label, valid_width_mask, word_positions]
        """
        out_enc = self.encoder(inputs)  # bsz c
        valid_width_masks = None
        word_positions = targets[-1]

        if len(targets) > 1:
            valid_width_masks = targets[-2]

        if self.training:
            label = targets[0]  # label
            final_out = self.decoder(
                inputs, out_enc, label, valid_width_masks, word_positions)
        else:
            final_out = self.decoder(
                inputs,
                out_enc,
                label=None,
                valid_width_masks=valid_width_masks,
                word_positions=word_positions,
                train_mode=False)
            # (bsz, seq_len, num_classes)

        return final_out
