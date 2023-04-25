from typing import List
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from . import mindcv_models


class MindCVBackboneWrapper(nn.Cell):
    '''
    It reuses the forward_features interface in mindcv models. Please check where the features are extracted.

    Note: text recognition models like CRNN expects output feature in shape [bs, c, h, w]. but some models in mindcv
    like ViT output features in shape [bs, c]. please check and pick accordingly.

    Args:
        pretrained (bool): Whether the model backbone is pretrained. Default; True
        checkpoint_path (str): The path of checkpoint files. Default: "".
        features_only (bool): Output the features at different strides instead. Default: False
        out_indices (list[int]): The indicies of the output features when `features_only` is `True`.
             Default: [0, 1, 2, 3, 4]

    Example:
        network = MindCVBackboneWrapper('resnet50', pretrained=True)
    '''
    def __init__(self, name, pretrained=True, ckpt_path=None, features_only: bool = False, out_indices: List[int] = [0, 1, 2, 3, 4], **kwargs):
        super().__init__()
        self.features_only = features_only

        model_name = name.replace('@mindcv', "").replace("mindcv.", "")
        network = mindcv_models.create_model(model_name, pretrained=pretrained, features_only=features_only, out_indices=out_indices)
        # for local checkpoint
        if ckpt_path is not None:
            checkpoint_param = load_checkpoint(ckpt_path)
            load_param_into_net(network, checkpoint_param)

        if not self.features_only:
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
        else:
            self.network = network
            self.out_channels = self.network.out_channels
            print(f'INFO: Load MindCV Backbone {model_name} with feature index {out_indices}, output channels: {self.out_channels}' )
            

    def construct(self, x):
        if self.features_only:
            features = self.network(x)
            return features
        else:
            features = self.network.forward_features(x)
            return [features]

if __name__=='__main__':
    model = MindCVBackboneWrapper('mindcv.resnet50', pretrained=False)
    x = ms.Tensor(np.random.rand(2, 3, 224, 224), dtype=ms.float32)
    ftr = model(x)
    print(ftr[0].shape)

    model = MindCVBackboneWrapper('mindcv.resnet50', pretrained=False, features_only=True, out_indices=[1, 2, 3, 4])
    x = ms.Tensor(np.random.rand(2, 3, 224, 224), dtype=ms.float32)
    ftr = model(x)
    for x in ftr:
        print(x.shape)
