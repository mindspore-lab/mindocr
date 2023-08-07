import random
from typing import Dict, List, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import ops

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None):
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return ms.Tensor(values).reshape(shape)


class ImageList(ms.Tensor):
    def __init__(self, tensor: ms.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> ms.Tensor:
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., :size[0], :size[1]]

    def astype(self, *args, **kwargs) -> "ImageList":
        cast_tensor = self.tensor.astype(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @staticmethod
    def from_tensors(
            tensors: List[ms.Tensor],
            size_divisibility: int = 0,
            pad_value: float = 0.0,
            padding_constraints: Optional[Dict[str, int]] = None,
    ) -> "ImageList":
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, ms.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [(np.array(x, np.int32)) for x in image_sizes]
        max_size = ms.Tensor(np.stack(image_sizes_tensor).max(0), dtype=ms.int32)

        if padding_constraints is not None:
            square_size = padding_constraints.get("square_size", 0)
            if square_size > 0:
                max_size[0] = max_size[1] = square_size
            if "size_divisibility" in padding_constraints:
                size_divisibility = padding_constraints["size_divisibility"]
        if size_divisibility > 1:
            stride = size_divisibility
            max_size = (max_size + (stride - 1)).floordiv(stride) * stride

        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            pad_op = ops.Pad(
                paddings=((0, 0), (0, 0), (padding_size[2], padding_size[3]), (padding_size[0], padding_size[1])))
            batched_imgs = pad_op(tensors[0].unsqueeze(0))
        else:
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size.asnumpy())
            batched_imgs = ms.Tensor(np.full(batch_shape, pad_value, dtype=np.float32), dtype=ms.float32)
            for i, img in enumerate(tensors):
                pad_img_op = ops.Pad(paddings=((0, 0), (0, 0), (0, img.shape[-2]), (0, img.shape[-1])))
                pad_img = pad_img_op(img)
                batched_imgs[i, ..., :img.shape[-2], :img.shape[-1]] = pad_img

        return ImageList(batched_imgs, image_sizes)


def random_attention_mask(shape, rng=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None)
    attn_mask[:, -1] = 1
    return attn_mask


def prepare_input():
    batch_size = 2
    image_size = 4
    num_channels = 3
    seq_length = 7
    vocab_size = 99
    range_bbox = 1000
    type_vocab_size = 16
    use_input_mask = True
    use_token_type_ids = True
    # num_labels = 3
    # type_sequence_label_size = 2
    # use_labels = True

    input_ids = ids_tensor((batch_size, seq_length), vocab_size)
    bbox = ids_tensor((batch_size, seq_length, 4), range_bbox)
    # Ensure that bbox is legal
    for i in range(bbox.shape[0]):
        for j in range(bbox.shape[1]):
            if bbox[i, j, 3] < bbox[i, j, 1]:
                t = bbox[i, j, 3]
                bbox[i, j, 3] = bbox[i, j, 1]
                bbox[i, j, 1] = t
            if bbox[i, j, 2] < bbox[i, j, 0]:
                t = bbox[i, j, 2]
                bbox[i, j, 2] = bbox[i, j, 0]
                bbox[i, j, 0] = t

    image = ops.zeros((batch_size, num_channels, image_size, image_size))

    input_mask = None
    if use_input_mask:
        input_mask = random_attention_mask((batch_size, seq_length))

    token_type_ids = None
    if use_token_type_ids:
        token_type_ids = ids_tensor((batch_size, seq_length), type_vocab_size)

    # sequence_labels = None
    # token_labels = None
    # if use_labels:
    #     sequence_labels = ids_tensor((batch_size,), type_sequence_label_size)
    #     token_labels = ids_tensor((batch_size, seq_length), num_labels)

    fake_input = {
        "bbox": bbox,
        "input_ids": input_ids,
        "image": image,
        "attention_mask": input_mask,
        "token_type_ids": token_type_ids
    }
    return fake_input
