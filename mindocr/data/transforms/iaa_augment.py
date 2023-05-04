import numpy as np
import imgaug
import imgaug.augmenters as iaa

__all__ = ['IaaAugment']


class IaaAugment:
    def __init__(self, **augments):
        aug_list = []
        for name, args in augments.items():
            if name == 'Affine':    # quick fix. In the future, dependency on imgaug will be deleted
                p = args.pop('p', 0.5)
                aug_list.append(iaa.Sometimes(p, iaa.Affine(**args)))
            else:
                aug_list.append(getattr(iaa, name)(**args))
        self._augmenter = iaa.Sequential(aug_list)

    def __call__(self, data):
        aug = self._augmenter.to_deterministic()    # to augment an image and its keypoints identically
        if 'polys' in data:
            new_polys = []
            for poly in data['polys']:
                kps = imgaug.KeypointsOnImage([imgaug.Keypoint(p[0], p[1]) for p in poly], shape=data['image'].shape)
                kps = aug.augment_keypoints(kps)
                new_polys.append(np.array([[kp.x, kp.y] for kp in kps.keypoints]))

            data['polys'] = np.array(new_polys) if isinstance(data['polys'], np.ndarray) else new_polys
        data['image'] = aug.augment_image(data['image'])

        return data
