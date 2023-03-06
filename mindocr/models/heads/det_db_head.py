import mindspore.nn as nn
from mindspore.common.initializer import HeNormal
from mindspore.common import initializer as init
from mindspore import ops

class DBHead(nn.Cell):
    def __init__(self, in_channels, k=50,
                 bias=False, adaptive=True, serial=False, training=True):
        super().__init__()

        self.k = k
        self.adaptive = adaptive
        self.training = training

        self.binarize = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, in_channels // 4, 2, stride=2, has_bias=True),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, 1, 2, stride=2, has_bias=True),
            nn.Sigmoid())

        self.weights_init(self.binarize)


        if adaptive:
            self.thresh = self._init_thresh(in_channels, serial=serial, bias=bias)
            self.weights_init(self.thresh)

    def weights_init(self, c):
        for m in c.cells():
            if isinstance(m, nn.Conv2dTranspose):
                m.weight = init.initializer(HeNormal(), m.weight.shape)
                m.bias = init.initializer('zeros', m.bias.shape)

            elif isinstance(m, nn.Conv2d):
                m.weight = init.initializer(HeNormal(), m.weight.shape)

            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def _init_thresh(self, in_channels, serial=False, bias=False):
        in_channels = in_channels
        if serial:
            in_channels += 1
        self.thresh = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, in_channels // 4, 2, stride=2, has_bias=True),
            # size * 2
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, 1, 2, stride=2, has_bias=True),
            nn.Sigmoid())

        return self.thresh

    def construct(self, feature):
        '''
        Args:
            feature (tensor): feature tensor
        Returns:
            pred: A dict which contains predictions.
                thresh: The threshold prediction
                binary: The text segmentation prediction.
                thresh_binary: Value produced by `step_function(binary - thresh)`.
        '''
        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(feature)
        pred = dict()
        pred['binary'] = binary

        #if self.adaptive and self.training:
        # TODO: use binary or thresh to do inference
        if self.training:
            thresh = self.thresh(feature)
            pred['thresh'] = thresh
            pred['thresh_binary'] = self.step_function(binary, thresh)

        return pred 

    def step_function(self, x, y):
        """Get the binary graph through binary and threshold."""
        reciprocal = ops.Reciprocal()
        exp = ops.Exp()

        return reciprocal(1 + exp(-self.k * (x - y)))
