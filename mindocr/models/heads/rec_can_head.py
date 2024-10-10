"""
CAN_HEAD_MODULE
"""
import math
import mindspore as ms
from mindspore import nn
from mindspore import ops

ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)


class ChannelAtt(nn.Cell):
    """Channel Attention of the Counting Module"""
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.SequentialCell([
            nn.Dense(channel, channel // reduction),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel),
            nn.Sigmoid()
        ])

    def construct(self, x):
        b, c, _, _ = x.shape
        y = ops.reshape(self.avg_pool(x), (b, c))
        y = ops.reshape(self.fc(y), (b, c, 1, 1))
        return x * y


class CountingDecoder(nn.Cell):
    """Single Counting Module"""
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.trans_layer = nn.SequentialCell([
            nn.Conv2d(
                self.in_channel,
                512,
                kernel_size=kernel_size,
                pad_mode='pad',
                padding=kernel_size // 2,
                has_bias=False,
            ),
            nn.BatchNorm2d(512)
        ])

        self.channel_att = ChannelAtt(512, 16)

        self.pred_layer = nn.SequentialCell([
            nn.Conv2d(
                512,
                self.out_channel,
                kernel_size=1,
                has_bias=False,
            ),
            nn.Sigmoid()
        ])

    def construct(self, x, mask):
        b, _, h, w = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)

        if mask is not None:
            x = x * mask
        x = ops.reshape(x, (b, self.out_channel, -1))
        x1 = ops.sum(x, -1)

        return x1, ops.reshape(x, (b, self.out_channel, h, w))


class Attention(nn.Cell):
    """Attention Module"""
    def __init__(self, hidden_size, attention_dim):
        super().__init__()
        self.hidden = hidden_size
        self.attention_dim = attention_dim
        self.hidden_weight = nn.Dense(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(
            1,
            512,
            kernel_size=11,
            pad_mode='pad',
            padding=5,
            has_bias=False
        )
        self.attention_weight = nn.Dense(512, self.attention_dim, has_bias=False)
        self.alpha_convert = nn.Dense(self.attention_dim, 1)

    def construct(
            self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None
    ):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0, 2, 3, 1))
        query_expanded = ops.unsqueeze(ops.unsqueeze(query, 1), 2)
        alpha_score = ops.tanh(
            query_expanded
            + coverage_alpha
            + cnn_features_trans.permute(0, 2, 3, 1)
        )
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = ops.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (
            ops.unsqueeze(ops.unsqueeze(ops.sum(ops.sum(energy_exp, -1), -1), 1), 2) + 1e-10
        )
        alpha_sum = ops.unsqueeze(alpha, 1) + alpha_sum
        context_vector = ops.sum(
            ops.sum((ops.unsqueeze(alpha, 1) * cnn_features), -1), -1
        )

        return context_vector, alpha, alpha_sum


class PositionEmbeddingSine(nn.Cell):
    """Position Embedding Sine Module of the Attention Decoder"""
    def __init__(
            self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True when scale is provided")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, x, mask):
        y_embed = ops.cumsum(mask, 1, dtype=ms.float32)
        x_embed = ops.cumsum(mask, 2, dtype=ms.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.arange(self.num_pos_feats, dtype=ms.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.unsqueeze(x_embed, 3) / dim_t
        pos_y = ops.unsqueeze(y_embed, 3) / dim_t

        pos_x = ops.flatten(
            ops.stack(
                [ops.sin(pos_x[:, :, :, 0::2]), ops.cos(pos_x[:, :, :, 1::2])],
                axis=4,
            ),
            'C',
            start_dim=3,
        )
        pos_y = ops.flatten(
            ops.stack(
                [ops.sin(pos_y[:, :, :, 0::2]), ops.cos(pos_y[:, :, :, 1::2])],
                axis=4,
            ),
            'C',
            start_dim=3,
        )

        pos = ops.concat([pos_x, pos_y], axis=3)
        pos = ops.transpose(pos, (0, 3, 1, 2))
        return pos


class AttDecoder(nn.Cell):
    """Attention Decoder Module"""
    def __init__(
            self,
            ratio,
            is_train,
            input_size,
            hidden_size,
            encoder_out_channel,
            dropout,
            dropout_ratio,
            word_num,
            counting_decoder_out_channel,
            attention,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_channel = encoder_out_channel
        self.attention_dim = attention["attention_dim"]
        self.dropout_prob = dropout
        self.ratio = ratio
        self.word_num = word_num
        self.counting_num = counting_decoder_out_channel
        self.is_train = is_train

        self.init_weight = nn.Dense(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_attention = Attention(self.hidden_size, self.attention_dim)

        self.encoder_feature_conv = nn.Conv2d(
            self.out_channel,
            self.attention_dim,
            kernel_size=attention["word_conv_kernel"],
            pad_mode="pad",
            padding=attention["word_conv_kernel"] // 2,
        )

        self.word_state_weight = nn.Dense(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Dense(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Dense(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Dense(self.counting_num, self.hidden_size)
        self.word_convert = nn.Dense(self.hidden_size, self.word_num)

        if dropout:
            self.dropout = nn.Dropout(p=dropout_ratio)

    def construct(self, cnn_features, labels, counting_preds, images_mask, is_train=True):
        if is_train:
            _, num_steps = labels.shape
        else:
            num_steps = 36

        batch_size, _, height, width = cnn_features.shape
        images_mask = images_mask[:, :, :: self.ratio, :: self.ratio]

        word_probs = ops.zeros((batch_size, num_steps, self.word_num))
        word_alpha_sum = ops.zeros((batch_size, 1, height, width))

        hidden = self.init_hidden(cnn_features, images_mask)
        counting_context_weighted = self.counting_context_weight(counting_preds)
        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])
        cnn_features_trans = cnn_features_trans + pos

        word = ops.ones((batch_size, 1), dtype=ms.int64)
        word = ops.squeeze(word, axis=1)

        for i in range(num_steps):
            word_embedding = self.embedding(word)
            hidden = self.word_input_gru(word_embedding, hidden)
            word_context_vec, _, word_alpha_sum = self.word_attention(
                cnn_features,
                cnn_features_trans,
                hidden,
                word_alpha_sum,
                images_mask
            )

            current_state = self.word_state_weight(hidden)
            word_weight_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)

            if self.dropout_prob:
                word_out_state = self.dropout(
                    current_state
                    + word_weight_embedding
                    + word_context_weighted
                    + counting_context_weighted
                )
            else:
                word_out_state = (
                    current_state
                    + word_weight_embedding
                    + word_context_weighted
                    + counting_context_weighted
                )

            word_prob = self.word_convert(word_out_state)
            word_probs[:, i] = word_prob

            if self.is_train:
                word = labels[:, i]
            else:
                word = word_prob.argmax(1)
                word = ops.multiply(
                    word, labels[:, i]
                )

        return word_probs

    def init_hidden(self, features, feature_mask):
        """Used to initialize the hidden layer"""
        average = ops.sum(
            ops.sum(features * feature_mask, dim=-1), dim=-1
        ) / ops.sum((ops.sum(feature_mask, dim=-1)), dim=-1)
        average = self.init_weight(average)
        return ops.tanh(average)


class CANHead(nn.Cell):
    r"""The CAN model is an algorithm used to recognize
    handwritten mathematical formulas.
    CAN Network is based on
    `"When Counting Meets HMER: Counting-Aware Network
    for Handwritten Mathematical Expression Recognition"
    <https://arxiv.org/abs/2207.11463>`_ paper.

    Args:
        "in_channels": number of channels for the input feature.
        "out_channels": number of channels for the output feature.
        "ratio": the ratio used to downsample the feature map.
        "attdecoder", the parameters needed to build an AttDecoder:
            - "is_train": indicates whether the model is in training mode.
            - "input_size":eEnter the size.
            - "hidden_size": Hidden layer size.
            - "encoder_out_channel": number of channels for the encoder output feature.
            - "dropout": whether to use dropout.
            - "dropout_ratio": the ratio of dropout.
            - "word_num": number of words.
            - "counting_decoder_out_channel": counts the decoder's output channels.
            - "attention", the parameters needed to build an attention mechanism:
                - "attention_dim": the dimension of the attention mechanism.
                - "word_conv_kernel": the size of the lexical convolution kernel.
                
    Return: 
        "word_probs": word probability distribution.
        "counting_preds1": count prediction 1, the number of words
                predicted by the 3*3 convolution kernel.
        "counting_preds2": count prediction 2, the number of words
                predicted by the 5*5 convolution kernel.
        "counting_preds": the mean predicted by the above two counts.
        

    Example:
        >>> # init a CANHead network
        >>> in_channels = 684
        >>> out_channels = 111
        >>> ratio = 16
        >>> attdecoder_params = {
        >>>     'is_train': True,
        >>>     'input_size': 256,
        >>>     'hidden_size': 256,
        >>>     'encoder_out_channel': in_channels,
        >>>     'dropout': True,
        >>>     'dropout_ratio': 0.5,
        >>>     'word_num': 111,
        >>>     'counting_decoder_out_channel': out_channels,
        >>>     'attention': {
        >>>         'attention_dim': 512,
        >>>         'word_conv_kernel': 1
        >>>     }
        >>> }
        >>> model = CANHead(in_channels, out_channels, ratio, attdecoder_params)
    """
    def __init__(self, in_channels, out_channels, ratio, attdecoder):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.counting_decoder1 = CountingDecoder(
            self.in_channels, self.out_channels, 3
        )
        self.counting_decoder2 = CountingDecoder(
            self.in_channels, self.out_channels, 5
        )

        self.decoder = AttDecoder(ratio, **attdecoder)

        self.ratio = ratio

    def construct(self, x, *args):
        cnn_features = x
        images_mask = args[0][0]
        labels = args[0][1]

        counting_mask = images_mask[:, :, :: self.ratio, :: self.ratio]
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2

        word_probs = self.decoder(cnn_features, labels, counting_preds, images_mask)

        return {
            'word_probs': word_probs,  
            'counting_preds': counting_preds,  
            'counting_preds1': counting_preds1,  
            'counting_preds2': counting_preds2  
            }
