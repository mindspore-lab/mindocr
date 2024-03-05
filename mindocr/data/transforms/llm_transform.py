import albumentations as alb
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from mindocr.models.third_party.mindformers.mindformers.models.clip import CLIPImageProcessor
from mindocr.models.third_party.mindformers.mindformers.dataset import (
    BCHW2BHWC, BatchResize, BatchToTensor,
    BatchNormalize, BatchCenterCrop, BatchPILize
)


def alb_wrapper(transform):
    def f(im):
        img = transform(image=np.asarray(im))["image"]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    return f


image_processor_high = alb_wrapper(
    alb.Compose(
        [
            alb.Resize(1024, 1024),
            alb.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
)


class VaryCLIPImageProcessor(CLIPImageProcessor):
    def __init__(self, image_resolution=224):
        super(VaryCLIPImageProcessor, self).__init__(image_resolution=image_resolution)

    def preprocess(self, images, **kwargs):
        bchw2bhwc = BCHW2BHWC()
        batch_pilizer = BatchPILize()
        batch_resizer = BatchResize(self.image_resolution)
        batch_crop = BatchCenterCrop(self.image_resolution)
        batch_totensor = BatchToTensor()
        batch_normalizer = BatchNormalize()

        if not self._bhwc_check(images):
            images = bchw2bhwc(images)
        images = batch_pilizer(images)
        images = batch_resizer(images)
        images = batch_crop(images)
        images = batch_totensor(images)
        images = batch_normalizer(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return np.row_stack([np.expand_dims(item, axis=0) for item in images])
        if len(images.shape) == 4:
            return images
        return np.expand_dims(images, axis=0)


image_processor = VaryCLIPImageProcessor().preprocess
