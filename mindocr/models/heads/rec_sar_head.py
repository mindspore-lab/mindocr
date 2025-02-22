import mindspore as ms
from mindspore import nn, ops


class SAREncoder(nn.Cell):
    """
    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_drop_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 d_model=512,
                 d_enc=512,
                 mask=True,
                 **kwargs):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask

        # LSTM Encoder
        if enc_bi_rnn:
            bidirectional = True
        else:
            bidirectional = False
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            batch_first=True,
            dropout=enc_drop_rnn,
            bidirectional=bidirectional)
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Dense(encoder_rnn_out_size, encoder_rnn_out_size)
        self.max_pool = nn.MaxPool2d((1, 1), stride=1, padding=0)

    def construct(self, feat, img_metas=None):
        if img_metas is not None:
            assert len(img_metas[0]) == ops.shape(feat)[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-2]

        feat_v = self.max_pool(feat)
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        feat_v = ops.transpose(feat_v, input_perm=(0, 2, 1))  # bsz * W * C
        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        if valid_ratios is not None:
            valid_hf = []
            T = ops.shape(holistic_feat)[1]
            for i in range(ops.shape(valid_ratios)[0]):
                valid_step = ops.minimum(
                    T, ops.ceil(valid_ratios[i] * T).astype('int32')) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = ops.stack(valid_hf, axis=0)
        else:
            valid_hf = holistic_feat[:, -1, :]  # bsz * C
        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat


class BaseDecoder(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()

    def construct_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def construct_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def construct(self,
                  feat,
                  out_enc,
                  label=None,
                  img_metas=None,
                  train_mode=True):

        if self.training:
            return self.construct_train(feat, out_enc, label, img_metas)
        return self.construct_test(feat, out_enc, img_metas)


class ParallelSARDecoder(BaseDecoder):
    """
    Args:
        out_channels (int): Output class number.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_drop_rnn (float): Dropout of RNN layer in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        d_k (int): Dim of channels of attention module.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        mask (bool): If True, mask padding in feature map.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.
    """

    def __init__(
            self,
            out_channels,
            enc_bi_rnn=False,
            dec_bi_rnn=False,
            dec_drop_rnn=0.0,
            dec_gru=False,
            d_model=512,
            d_enc=512,
            d_k=64,
            pred_dropout=0.1,
            max_text_length=30,
            mask=True,
            pred_concat=True,
            **kwargs):
        super().__init__()

        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)

        # 2D attention layer
        self.conv1x1_1 = nn.Dense(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(
            d_model, d_k, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.conv1x1_2 = nn.Dense(d_k, 1)

        # Decoder RNN layer
        if dec_bi_rnn:
            bidirectional = True
        else:
            bidirectional = False

        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_drop_rnn,
            bidirectional=bidirectional)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.padding_idx)

        # Prediction layer
        self.pred_dropout = nn.Dropout(p=pred_dropout)
        pred_num_classes = self.num_classes - 1
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        self.prediction = nn.Dense(fc_in_channel, pred_num_classes)

    def _2d_attention(self,
                      decoder_input,
                      feat,
                      holistic_feat,
                      valid_width_masks=None):

        y = self.rnn_decoder(decoder_input)[0]
        # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.shape
        attn_query = ops.unsqueeze(attn_query, dim=3)
        attn_query = ops.unsqueeze(attn_query, dim=4)
        # (bsz, seq_len + 1, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)
        # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = ops.tanh(ops.add(attn_key, attn_query))

        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = ops.transpose(attn_weight, input_perm=(0, 1, 3, 4, 2))
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight)
        # bsz * (seq_len + 1) * h * w * 1
        bsz, T, h, w, c = ops.shape(attn_weight)
        assert c == 1

        if valid_width_masks is not None:
            attn_weight = ops.split(attn_weight, 1, axis=0)
            attn_weight_list = []
            for i, valid_width_mask in enumerate(valid_width_masks):
                attn_weight_i = attn_weight[i].squeeze(0)  # (T, h, w, c)
                attn_weight_i = attn_weight_i.transpose((0, 1, 3, 2))  # (T, h, c, w)
                attn_weight_i = attn_weight_i.view((-1, w))  # (T*h*c, w)
                Tch = T * h * c
                valid_width_mask = valid_width_mask.repeat_interleave(Tch, 0)  # (T*h*c, w)
                valid_width_mask = ops.cast(valid_width_mask, ms.bool_)
                attn_weight_i = ops.select(
                    valid_width_mask, attn_weight_i.astype(ms.float32), float('-inf'))  # (T*h*c, w)
                attn_weight_i = attn_weight_i.astype(ms.float16).view((T, h, c, w))  # (T, h, c, w)
                attn_weight_i = attn_weight_i.transpose((0, 1, 3, 2))  # (T, h, w, c)
                attn_weight_list.append(attn_weight_i)
            attn_weight = ops.concat(attn_weight_list, axis=0)  # (bsz, T, h, w, c)

        attn_weight = ops.reshape(attn_weight, [bsz, T, -1])
        attn_weight = ops.softmax(attn_weight, axis=-1)

        attn_weight = ops.reshape(attn_weight, [bsz, T, h, w, c])
        attn_weight = ops.transpose(attn_weight, input_perm=(0, 1, 4, 2, 3))
        # attn_weight: bsz * T * c * h * w
        # feat: bsz * c * h * w
        attn_feat = ops.sum(ops.multiply(feat.unsqueeze(1), attn_weight),
                            (3, 4),
                            keepdim=False)
        # bsz * (seq_len + 1) * C

        # Linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.shape[-1]
            holistic_feat = holistic_feat.broadcast_to((bsz, seq_len, hf_c))
            y = self.prediction(
                ops.concat((y, attn_feat.astype(y.dtype),
                            holistic_feat.astype(y.dtype)), 2))
        else:
            y = self.prediction(attn_feat)
        # bsz * (seq_len + 1) * num_classes
        if self.training:
            y = self.pred_dropout(y)

        return y

    def construct_train(self, feat, out_enc, label, img_metas):
        if img_metas is not None:
            assert ops.shape(img_metas[0])[0] == ops.shape(feat)[0]

        valid_width_masks = None
        if img_metas is not None and self.mask:
            valid_width_masks = img_metas[-1]

        lab_embedding = self.embedding(label)  # bsz * seq_len * emb_dim

        out_enc = out_enc.unsqueeze(1).astype(lab_embedding.dtype)  # bsz * 1 * emb_dim

        in_dec = ops.concat((out_enc, lab_embedding), axis=1)  # bsz * (seq_len + 1) * C

        out_dec = self._2d_attention(
            in_dec, feat, out_enc, valid_width_masks=valid_width_masks)

        return out_dec[:, 1:, :]  # bsz * seq_len * num_classes

    def construct_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_width_masks = None
        if img_metas is not None and self.mask:
            valid_width_masks = img_metas[-1]

        seq_len = self.max_seq_len
        bsz = feat.shape[0]
        start_token = ops.full(
            (bsz, ), fill_value=self.start_idx, dtype=ms.int32)  # bsz

        start_token = self.embedding(start_token)  # bsz * emb_dim

        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1)

        start_token = start_token.broadcast_to((bsz, seq_len, emb_dim))  # bsz * seq_len * emb_dim

        out_enc = out_enc.unsqueeze(1)  # bsz * 1 * emb_dim

        decoder_input = ops.concat((out_enc, start_token), axis=1)  # bsz * (seq_len + 1) * emb_dim

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(
                decoder_input, feat, out_enc, valid_width_masks=valid_width_masks)
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            char_output = ops.softmax(char_output, -1)
            outputs.append(char_output)
            max_idx = ops.argmax(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        outputs = ops.stack(outputs, 1)  # bsz * seq_len * num_classes

        return outputs


class SARHead(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 enc_dim=512,
                 max_text_length=30,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 dec_bi_rnn=False,
                 dec_drop_rnn=0.0,
                 dec_gru=False,
                 d_k=512,
                 pred_dropout=0.1,
                 pred_concat=True,
                 **kwargs):
        super(SARHead, self).__init__()

        # encoder module
        self.encoder = SAREncoder(
            enc_bi_rnn=enc_bi_rnn,
            enc_drop_rnn=enc_drop_rnn,
            enc_gru=enc_gru,
            d_model=in_channels,
            d_enc=enc_dim)

        # decoder module
        self.decoder = ParallelSARDecoder(
            out_channels=out_channels,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_model=in_channels,
            d_enc=enc_dim,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat)

    def construct(self, feat, targets=None):
        holistic_feat = self.encoder(feat, targets)  # bsz c

        if self.training:
            label = targets[0]  # label
            final_out = self.decoder(
                feat, holistic_feat, label, img_metas=targets)
        else:
            final_out = self.decoder(
                feat,
                holistic_feat,
                label=None,
                img_metas=targets,
                train_mode=False)
            # (bsz, seq_len, num_classes)

        return final_out
