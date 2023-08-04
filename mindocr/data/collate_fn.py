import numpy as np

__all__ = ["SSLRotateCollate"]


class SSLRotateCollate:
    """Collate [(4x3xHxW), (4,)]"""

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle

    def __call__(self, image, label, batch_info):
        image = np.concatenate(image, axis=0)
        label = np.concatenate(label, axis=0)

        if self.shuffle:
            inds = np.arange(image.shape[0])
            np.random.shuffle(inds)
            image = image[inds]
            label = label[inds]
        return image, label
