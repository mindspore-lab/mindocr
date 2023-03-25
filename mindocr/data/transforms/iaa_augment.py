"""
iaa transform
"""
import numpy as np
import imgaug
import imgaug.augmenters as iaa

__all__ = ['IaaAugment']

def compose_iaa_transforms(args, root=True):
    if (args is None) or (len(args) == 0):
        return None
    elif isinstance(args, list):
        if root:
            sequence = [compose_iaa_transforms(value, root=False) for value in args]
            return iaa.Sequential(sequence)
        else:
            return getattr(iaa, args[0])( *[_list_to_tuple(a) for a in args[1:]])
    elif isinstance(args, dict):
        cls = getattr(iaa, args['type'])
        return cls(**{
            k: _list_to_tuple(v)
            for k, v in args['args'].items()
        })
    else:
        raise ValueError('Unknown augmenter arg: ' + str(args))


def _list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(obj)
    return obj


class IaaAugment():
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'Resize',
                'args': {
                    'size': [0.5, 3]
                }
            }]
        self.augmenter = compose_iaa_transforms(augmenter_args)

    def __call__(self, data):
        image = data['image']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly