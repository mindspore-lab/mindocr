import os
import sys
import numpy as np

mindformers_path = os.path.abspath(os.path.join(os.path.dirname(file),
                                   "../../../mindocr/models/third_party/mindformers"))
sys.path.insert(0, mindformers_path)

mindpet_path = os.path.abspath(os.path.join(os.path.dirname(file),
                               "../../../mindocr/models/third_party/mindpet"))
sys.path.insert(0, mindpet_path)

import mindspore as ms
from mindspore import nn, load_checkpoint, load_param_into_net

from mindformers.models import SAMImageEncoder, ImageEncoderConfig


class SAMEncoder(SAMImageEncoder):
    """SAM encoder for Vary system"""
    def init(self, config) -> None:
        super().init(config)
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = super().construct(x)
        x = self.net_2(x)
        x = self.net_3(x)
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        return x

if __name__ == "__main__":
    """
    realize following torch code in vary_qwen_vary.py using MindSpore:

    with torch.set_grad_enabled(False):
        cnn_feature = vision_tower_high(image[1])
        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)
    """
    sam_config_dict = dict(img_size=1024,  # img_size in ImageEncoderViT
                        patch_size=16,  # patch_size in ImageEncoderViT
                        in_chans=3,  # in_chans in ImageEncoderViT
                        embed_dim=768,  # encoder_embed_dim in build_sam_vit_b
                        depth=12,  # encoder_depth in build_sam_vit_b
                        num_heads=12,  # encoder_num_heads in build_sam_vit_b
                        mlp_ratio=4,  # mlp_ratio in ImageEncoderViT
                        out_chans=256,  # out_chans in ImageEncoderViT
                        qkv_bias=True,  # qkv_bias in ImageEncoderViT
                        layer_norm_eps=1.e-6,  # refer to Vary-master\vary\model\vision_encoder\sam.py
                        use_abs_pos=True,  # use_abs_pos in ImageEncoderViT
                        use_rel_pos=True,  # use_rel_pos in ImageEncoderViT
                        window_size=14,
                        global_attn_indexes=[2, 5, 8, 11],  # encoder_global_attn_indexes in build_sam_vit_b
                        compute_dtype="float32",
                        layernorm_compute_type="float32",
                        softmax_compute_type="float32",
                        param_init_type="float32")
    img = np.load("/data0/perf/git/mindocr/tests/ut/sam_img.npy").astype(np.float32)
    img_ts = ms.Tensor.from_numpy(img)

    sam_config = ImageEncoderConfig(**sam_config_dict)
    sam_inst = SAMEncoder(sam_config)

    ckpt_dict = load_checkpoint("/data0/perf/code/wtc_ms/Vary/Vary-master/vary_toy_weights/ms_sam.ckpt")
    load_param_into_net(sam_inst, ckpt_dict)
    sam_inst.set_train(False)

    cnn_feature = sam_inst(img_ts)

    print("sam test complete")
