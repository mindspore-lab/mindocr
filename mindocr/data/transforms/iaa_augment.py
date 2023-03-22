import numpy as np
import imgaug
import imgaug.augmenters as iaa

__all__ = ['IaaAugment']


class IaaAugment:
    def __init__(self, **augments):
        self._augmenter = iaa.Sequential([getattr(iaa, aug)(**args) for aug, args in augments.items()])

    def __call__(self, data):
        aug = self._augmenter.to_deterministic()    # to augment an image and its keypoints identically
        data['image'] = aug.augment_image(data['image'])

        if 'polys' in data:
            new_polys = []
            for poly in data['polys']:
                kps = imgaug.KeypointsOnImage([imgaug.Keypoint(p[0], p[1]) for p in poly], shape=data['image'].shape)
                kps = aug.augment_keypoints(kps)
                new_polys.append(np.array([[kp.x, kp.y] for kp in kps.keypoints]))

            data['polys'] = np.array(new_polys) if isinstance(data['polys'], np.ndarray) else new_polys

        return data
