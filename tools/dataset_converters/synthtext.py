import itertools
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from shapely.geometry import Polygon, box
from tqdm import tqdm

from mindocr.data.utils.polygon_utils import sort_clockwise
from tools.dataset_converters.utils.lmdb_writer import create_lmdb_dataset
from tools.infer.text.utils import crop_text_region


class SYNTHTEXT_Converter:
    """
    Validate polygons and sort vertices in SynthText dataset. The filtered dataset will be stored
    in the same format as the original one for compatibility purposes.

    Args:
        min_area: area below which polygons will be filtered out
    """

    def __init__(self, *args, **kwargs):
        self._image_dir = None

    def _sort_and_validate(self, sample: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
        """
        Sort vertices in clockwise order (to eliminate self-intersection) and filter invalid polygons out.
        Args:
            sample: tuple containing polygons and texts instances.
        Returns:
            filtered polygons and texts.
        """
        path, polys, texts = sample
        polys = polys.transpose().reshape(-1, 4, 2)  # some labels have (4, 2) shape (no batch dimension)
        texts = [t for text in texts.tolist() for t in text.split()]  # TODO: check the correctness of texts order
        size = np.array(Image.open(os.path.join(self._image_dir, path.item())).size) - 1  # (w, h)
        border = box(0, 0, *size)

        # SynthText has a lot of mistakes in the dataset that may affect the data processing pipeline
        # Sort vertices clockwise and filter invalid polygons out
        new_polys, new_texts = [], []
        for np_poly, text in zip(polys, texts):
            # fix self-intersection by sorting vertices
            np_poly = sort_clockwise(np_poly)
            # check if the polygon is valid and lies within the visible borders
            poly = Polygon(np_poly)
            if poly.is_valid and poly.intersects(border):
                np_poly = np.clip(np_poly, 0, size)  # clip bbox to be within the visible region
                poly = Polygon(np_poly)  # check the polygon validity once again after clipping
                if poly.is_valid and not poly.equals(border):
                    new_polys.append(np_poly)
                    new_texts.append(text)

        return np.array(new_polys).transpose(), np.array(new_texts)  # preserve polygons' axes order

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        if task == "det":
            self.convert_det(image_dir, label_path, output_path, save_output=True)
        elif task == "rec_lmdb":
            self.convert_rec_lmdb(image_dir, label_path, output_path)
        else:
            raise ValueError(f"Unsupported task `{task}`.")

    def convert_det(self, image_dir=None, label_path=None, output_path=None):
        self._image_dir = image_dir
        print("Loading SynthText dataset. It might take a while...")
        mat = loadmat(label_path)

        # use multiprocessing to process the dataset faster
        with ProcessPoolExecutor(max_workers=8) as pool:
            data_list = list(
                tqdm(
                    pool.map(self._sort_and_validate, zip(mat["imnames"][0], mat["wordBB"][0], mat["txt"][0])),
                    total=len(mat["imnames"][0]),
                    desc="Processing data",
                    miniters=10000,
                )
            )

        wordBB, txt = zip(*data_list)
        for i in range(len(mat["wordBB"][0])):  # how to stack wordBB?
            mat["wordBB"][0][i] = wordBB[i]
        mat["txt"] = np.array(txt).reshape(1, -1)

        print("Saving...")
        savemat(
            output_path,
            {
                "charBB": mat["charBB"],  # save as it is
                "wordBB": mat["wordBB"],
                "imnames": mat["imnames"],
                "txt": mat["txt"],
            },
            do_compression=True,
        )

    def _crop_with_single_text(self, sample: Tuple[str, Dict[str, Any]]) -> Tuple[bytes, str]:
        images = []
        labels = []

        img_path, img_info = sample
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"WARNING: {str(e)}")
            return images, labels

        for item_info in img_info:
            try:
                sub_img = crop_text_region(img, item_info["poly"], box_type="poly", rotate_if_vertical=False)
                # encode image as JPEG format
                sub_img = cv2.imencode(".jpg", sub_img)[1].tobytes()
            except Exception as e:
                print(f"WARNING: {str(e)}")
                continue

            sub_text = item_info["text"]
            images.append(sub_img)
            labels.append(sub_text)
        return images, labels

    def convert_rec_lmdb(self, image_dir=None, label_path=None, output_path=None):
        self._image_dir = image_dir
        print("Loading SynthText dataset. It might take a while...")
        mat = loadmat(label_path)

        # use multiprocessing to process the dataset faster
        with ProcessPoolExecutor(max_workers=8) as pool:
            data_list = list(
                tqdm(
                    pool.map(self._sort_and_validate, zip(mat["imnames"][0], mat["wordBB"][0], mat["txt"][0])),
                    total=len(mat["imnames"][0]),
                    desc="Processing data",
                )
            )

        wordBB, txt = zip(*data_list)
        for i in range(len(mat["wordBB"][0])):
            mat["wordBB"][0][i] = wordBB[i]
        mat["txt"] = np.array(txt).reshape(1, -1)

        data_list = defaultdict(list)
        for image, polys, texts in zip(mat["imnames"][0], mat["wordBB"][0], mat["txt"][0]):
            texts = [t for text in texts.tolist() for t in text.split()]
            polys = polys.transpose().reshape(-1, 4, 2)
            img_path = os.path.join(image_dir, image.item())
            for poly, text in zip(polys, texts):
                data_list[img_path].append(
                    {
                        "poly": poly,
                        "text": text,
                    }
                )

        with ProcessPoolExecutor(max_workers=8) as pool:
            data_list = list(
                tqdm(
                    pool.map(self._crop_with_single_text, zip(data_list.keys(), data_list.values())),
                    total=len(data_list),
                    desc="Cropping data",
                )
            )

        images, labels = zip(*data_list)
        images = iter(itertools.chain(*images))
        labels = iter(itertools.chain(*labels))

        print("Creating the LMDB dataset.")
        create_lmdb_dataset(images, labels, output_path=output_path)
