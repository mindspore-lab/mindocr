import numpy as np
import imgaug
import imgaug.augmenters as iaa

__all__ = ['IaaAugment']


class IaaAugment:
    def __init__(self, **augments):
        aug_list = []
        # assume Augment is no applied in test time. TODO: consider test time augmentation.
        if 'is_train' in augments:
            self.is_train = augments.pop('is_train')

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
            if self.is_train:
                new_polys = []
                for poly in data['polys']:
                    kps = imgaug.KeypointsOnImage([imgaug.Keypoint(p[0], p[1]) for p in poly], shape=data['image'].shape)
                    kps = aug.augment_keypoints(kps)
                    new_polys.append(np.array([[kp.x, kp.y] for kp in kps.keypoints]))

                data['polys'] = np.array(new_polys) if isinstance(data['polys'], np.ndarray) else new_polys
            else:
                raise ValueError('Test time augmentation is not supported for detection currently (due to transformed polygons can be not recoveried for evaluation). IaaAugment should only be used for training.')

        data['image'] = aug.augment_image(data['image'])

        return data
