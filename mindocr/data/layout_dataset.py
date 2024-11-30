import glob
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["PublayNetDataset"]


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


class PublayNetDataset:
    """
    Load the PublayNet dataset (yolo format publaynet labels)

    Args:
        dataset_path (str): dataset label directory for dataset.
        for example:
            PUBLAYNET_ROOT
                ├── train.txt
                ├── val.txt
                ├── train.json
                ├── val.json
                ├── train
                │   ├── PMC5090051_00006.jpg
                │   ├── PMC5090051_00006.txt
                │   ├── PMC4494574_00001.jpg
                │   └── PMC4494574_00001.txt
                └── val
                    ├── PMC5015546_00000.jpg
                    ├── PMC5015546_00000.txt
                    ├── PMC3973169_00004.jpg
                    └── PMC3973169_00004.txt
        dataset_path (str): ./publaynet/train.txt
        transform_pipeline (list): A list of images data enhancements
            that apply data enhancements on data set objects in order.
    """

    def __init__(
        self,
        dataset_path="",
        annotations_path="",
        img_size=640,
        transform_pipeline=None,
        is_train=False,
        augment=False,
        rect=False,
        batch_size=32,
        stride=32,
        pad=0.0,
        **kwargs,
    ):
        self.cache_version = 0.1
        self.path = dataset_path

        # acceptable image suffixes
        self.img_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
        self.help_url = "https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"

        self.model_name = kwargs.get("model_name", "")
        self.img_size = img_size
        self.augment = augment
        self.rect = rect
        self.stride = stride
        self.transform_pipeline = transform_pipeline
        self.is_train = is_train
        if is_train:
            self.dataset_column_names = ["images", "labels"]
        else:
            self.dataset_column_names = ["images", "labels", "image_ids", "hw_ori", "hw_scale", "pad"]

        try:
            f = []  # image files
            for p in self.path if isinstance(self.path, list) else [self.path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p, "r") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                else:
                    raise Exception(f"{p} does not exist")
            self.img_files = sorted([x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in self.img_formats])
            assert self.img_files, "No images found"
        except Exception as e:
            raise Exception(f"Error loading data from {self.path}: {e}\nSee {self.help_url}")

        # Check cache
        self.label_files = self._img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache.npy")  # cached labels
        if cache_path.is_file():
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            if cache["version"] == self.cache_version and cache["hash"] == self._get_hash(
                self.label_files + self.img_files
            ):
                logger.info("Dataset Cache file hash/version check success.")
                logger.info(f"Load dataset cache from [{cache_path}] success.")
            else:
                logger.info("Dataset cache file hash/version check fail.")
                logger.info("Dataset caching now...")
                cache, exists = self.cache_labels(cache_path), False  # cache
                logger.info("Dataset caching success.")
        else:
            logger.info("No dataset cache available, caching now...")
            cache, exists = self.cache_labels(cache_path), False  # cache
            logger.info("Dataset caching success.")

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f"No labels in {cache_path}. Can not train without labels. See {self.help_url}"

        # Read cache
        cache.pop("hash")  # remove hash
        cache.pop("version")  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.img_shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        if not is_train:
            if os.path.isfile(annotations_path):
                with open(annotations_path, "r") as f:
                    data = json.load(f)
                file_id_dict = dict()
                for item in data["images"]:
                    file_id_dict[item["file_name"]] = item["id"]
                self.image_ids = [file_id_dict[img_file.split("/")[-1]] for img_file in self.img_files]
            else:
                self.image_ids = self.img_files
        else:
            self.image_ids = None
        self.label_files = self._img2label_paths(cache.keys())  # update

        n = len(labels)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int_)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_hw_ori, self.indices = (
            [
                None,
            ]
            * n,
            [
                None,
            ]
            * n,
            range(n),
        )

        # Rectangular Train/Test
        if self.rect:
            # Sort by aspect ratio
            s = self.img_shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.img_shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride

    def cache_labels(self, path=Path("./labels.cache")):
        # Get orientation exif tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc="Scanning images", total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = self._exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                assert im.format.lower() in self.img_formats, f"invalid image format {im.format}"

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, "r") as f:
                        ll = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in ll]):  # is segment
                            classes = np.array([x[0] for x in ll], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in ll]  # (cls, xy1...)
                            ll = np.concatenate(
                                (classes.reshape(-1, 1), self._segments2boxes(segments)), 1
                            )  # (cls, xywh)
                        ll = np.array(ll, dtype=np.float32)
                    if len(ll):
                        assert ll.shape[1] == 5, "labels require 5 columns each"
                        assert (ll >= 0).all(), "negative labels"
                        assert (ll[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels"
                        assert np.unique(ll, axis=0).shape[0] == ll.shape[0], "duplicate labels"
                    else:
                        ne += 1  # label empty
                        ll = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    ll = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [ll, shape, segments]
            except Exception as e:
                nc += 1
                print(f"WARNING: Ignoring corrupted image and/or label {im_file}: {e}")

            pbar.desc = (
                f"Scanning '{path.parent / path.stem}' images and labels... "
                f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            )
        pbar.close()

        if nf == 0:
            print(f"WARNING: No labels found in {path}. See {self.help_url}")

        x["hash"] = self._get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, i + 1
        x["version"] = self.cache_version  # cache version
        np.save(path, x)  # save for next time
        logger.info(f"New cache created: {path}")
        return x

    def __getitem__(self, index):
        index, image, labels, hw_ori, hw_scale, pad = index, None, None, None, None, None
        for _i, ori_trans in enumerate(self.transform_pipeline):
            _trans = ori_trans.copy()
            func_name, prob = _trans.pop("func_name"), _trans.pop("prob", 1.0)
            if random.random() < prob:
                if func_name == "letterbox":
                    image, hw_ori = self.load_image(index)
                    labels = self.labels[index].copy()
                    new_shape = self.img_size if not self.rect else self.batch_shapes[self.batch[index]]
                    image, labels, hw_ori, hw_scale, pad = self.letterbox(image, labels, hw_ori, new_shape, **_trans)
                else:
                    if image is None:
                        image, hw_ori = self.load_image(index)
                        labels = self.labels[index].copy()
                        new_shape = self.img_size if not self.rect else self.batch_shapes[self.batch[index]]
                        image, labels, hw_ori, hw_scale, pad = self.letterbox(
                            image,
                            labels,
                            hw_ori,
                            new_shape,
                        )
                    image, labels = getattr(self, func_name)(image, labels, **_trans)

        image = np.ascontiguousarray(image)

        if self.is_train:
            return image, labels
        else:
            return image, labels, self.image_ids[index], hw_ori, hw_scale, pad

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, "Image Not Found " + path
            h_ori, w_ori = img.shape[:2]  # orig hw
            if self.model_name == "layoutlmv3":
                r = self.img_size / min(h_ori, w_ori)
                img = cv2.resize(img, (int(round(w_ori * r)), int(round(h_ori * r))), interpolation=cv2.INTER_LINEAR)
            else:
                r = self.img_size / max(h_ori, w_ori)  # resize image to img_size
                if r != 1:  # always resize down, only resize up if training with augmentation
                    interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                    img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)

            return img, np.array([h_ori, w_ori])  # img, hw_original
        else:
            return self.imgs[index], self.img_hw_ori[index]  # img, hw_original

    def load_samples(self, index):
        # loads images in a 4-mosaic
        labels4, segments4 = [], []
        s = self.img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _ = self.load_image(index)
            (h, w) = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        sample_labels, sample_images, sample_masks = self._sample_segments(img4, labels4, segments4, probability=0.5)

        return sample_labels, sample_images, sample_masks

    def letterbox(self, image, labels, hw_ori, new_shape, scaleup=False, color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        h, w = shape[:]
        h0, w0 = hw_ori
        hw_scale = np.array([h / h0, w / w0])
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        if self.model_name == "layoutlmv3":
            r = max(new_shape[0] / shape[0], new_shape[1] / shape[1])
        else:
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = abs(new_shape[1] - new_unpad[0]), abs(new_shape[0] - new_unpad[1])  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        hw_pad = np.array([dh, dw])

        if self.model_name != "layoutlmv3":
            if shape[::-1] != new_unpad:  # resize
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # convert labels
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], r * w, r * h, padw=hw_pad[1], padh=hw_pad[0])

        return image, labels, hw_ori, hw_scale, hw_pad

    def label_norm(self, image, labels, xyxy2xywh_=True):
        if len(labels) == 0:
            return image, labels

        if xyxy2xywh_:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

        labels[:, [2, 4]] /= image.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= image.shape[1]  # normalized width 0-1

        return image, labels

    def label_pad(self, image, labels, padding_size=160, padding_value=-1):
        # create fixed label, avoid dynamic shape problem.
        labels_out = np.full((padding_size, 6), padding_value, dtype=np.float32)
        nL = len(labels)
        if nL:
            labels_out[: min(nL, padding_size), 0:1] = 0.0
            labels_out[: min(nL, padding_size), 1:] = labels[: min(nL, padding_size), :]
        return image, labels_out

    def image_norm(self, image, labels, scale=255.0):
        image = image.astype(np.float32, copy=False)
        image /= scale
        return image, labels

    def image_normal(self, image, labels, mean, std):
        image = (image - mean) / std
        image = image.astype(np.float32, copy=False)
        return image, labels

    def image_batch_pad(self, image, labels, max_size=None):
        image_size = image.shape[-2:]
        if max_size is None:
            max_size = np.array(image_size)
        else:
            max_size = np.array([max_size, max_size])

        h_pad = int(max_size[0] - image_size[0])
        w_pad = int(max_size[1] - image_size[1])

        padding_size = ((0, 0), (0, h_pad), (0, w_pad))
        batched_imgs = np.pad(image, padding_size, mode="constant", constant_values=0)
        return batched_imgs, labels

    def image_transpose(self, image, labels, bgr2rgb=True, hwc2chw=True):
        if bgr2rgb:
            image = image[:, :, ::-1]
        if hwc2chw:
            image = image.transpose(2, 0, 1)
        return image, labels

    def _sample_segments(self, img, labels, segments, probability=0.5):
        # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
        n = len(segments)
        sample_labels = []
        sample_images = []
        sample_masks = []
        if probability and n:
            h, w, c = img.shape  # height, width, channels
            for j in random.sample(range(n), k=round(probability * n)):
                ll = labels[j]
                box = (
                    ll[1].astype(int).clip(0, w - 1),
                    ll[2].astype(int).clip(0, h - 1),
                    ll[3].astype(int).clip(0, w - 1),
                    ll[4].astype(int).clip(0, h - 1),
                )

                if (box[2] <= box[0]) or (box[3] <= box[1]):
                    continue

                sample_labels.append(ll[0])

                mask = np.zeros(img.shape, np.uint8)

                cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
                sample_masks.append(mask[box[1] : box[3], box[0] : box[2], :])

                result = cv2.bitwise_and(src1=img, src2=mask)
                i = result > 0  # pixels to replace
                mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
                sample_images.append(mask[box[1] : box[3], box[0] : box[2], :])

        return sample_labels, sample_images, sample_masks

    def _img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep  # /images/, /labels/ substrings
        return ["txt".join(x.replace(sa, sb, 1).rsplit(x.split(".")[-1], 1)) for x in img_paths]

    def _get_hash(self, paths):
        # Returns a single hash value of a list of paths (files or dirs)
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
        h = hashlib.md5(str(size).encode())  # hash sizes
        h.update("".join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash

    def _exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except Exception as e:
            error_log = f"no need rotation when [{e}] happen"
            if "NoneType" not in error_log:
                print(error_log)

        return s

    def _segments2boxes(self, segments):
        # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
        boxes = []
        for s in segments:
            x, y = s.T  # segment xy
            boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
        return xyxy2xywh(np.array(boxes))  # cls, xywh

    def get_output_columns(self):
        return self.dataset_column_names


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y
