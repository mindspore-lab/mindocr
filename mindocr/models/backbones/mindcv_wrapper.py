import numpy as np
from . import mindcv_models
import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net

class MindCVBackboneWrapper(nn.Cell):
    '''
    It reuses the forward_features interface in mindcv models. Please check where the features are extracted.
    Only support forward the feature of the last layer of the backbone currently.

    Note: text recognition models like CRNN expects output feature in shape [bs, c, h, w]. but some models in mindcv
    like ViT output features in shape [bs, c]. please check and pick accordingly.

    Example:
        network = MindCVBackboneWrapper('resnet50', pretrained=True)
    '''
    def __init__(self, name, pretrained=True, ckpt_path=None, **kwargs):
        super().__init__()

        #self.out_indices = out_indices
        #self.out_channels =[ch*block.expansion for ch in [64, 128, 256, 512]]

        model_name = name.replace('@mindcv', "").replace("mindcv.", "")
        network = mindcv_models.create_model(model_name, pretrained=pretrained)
        # for local checkpaoint
        if ckpt_path is not None:
            checkpoint_param = load_checkpoint(ckpt_path)
            load_param_into_net(network, checkpoint_param)

        # TODO: add include_top or feature_only param in mindcv
        if hasattr(network, 'classifier'):
            del network.classifier  # remove the original header to avoid confusion

        self.network = network
        # probe to get out_channels
        #network.eval()
        # TODO: get image input size from default cfg
        x = ms.Tensor(np.random.rand(2, 3, 224, 224), dtype=ms.float32)
        h = network.forward_features(x)
        h = ops.stop_gradient(h)
        self.out_channels = h.shape[1]

        print(f'INFO: Load MindCV Backbone {model_name}, the output features shape for input 224x224 is {h.shape}. \n\tProbed out_channels : ', self.out_channels )

    def construct(self, x):
        features = self.network.forward_features(x)

        return [features]

if __name__=='__main__':
    #model = MindCVBackboneWrapper('mindcv.vit_b_32_224', pretrained=False)
    model = MindCVBackboneWrapper('mindcv.resnet50', pretrained=False)
    x = ms.Tensor(np.random.rand(2, 3, 224, 224), dtype=ms.float32)
    ftr = model(x)
    print(ftr[0].shape)
