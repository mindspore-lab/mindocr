import numpy as np
import imgaug.augmenters as iaa
from imgaug import Keypoint, KeypointsOnImage

__all__ = ['IaaAugment']


class IaaAugment:
    def __init__(self, polygons=True, **augments):
        self._augmenter = iaa.Sequential([getattr(iaa, aug)(**args) for aug, args in augments.items()])
        self.output_columns = ['image', 'polys'] if polygons else ['image']

    def __call__(self, data):
        aug = self._augmenter.to_deterministic()    # to augment an image and its keypoints identically
        original_shape = data['image'].shape
        data['image'] = aug.augment_image(data['image'])

        if 'polys' in self.output_columns:
            new_polys = []
            for poly in data['polys']:
                kps = KeypointsOnImage([Keypoint(p[0], p[1]) for p in poly], shape=original_shape)
                kps = aug.augment_keypoints(kps)
                new_polys.append(np.array([[kp.x, kp.y] for kp in kps.keypoints]))

            data['polys'] = np.array(new_polys) if isinstance(data['polys'], np.ndarray) else new_polys

        return data
