import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from shapely.geometry import Polygon, box
from tqdm import tqdm

from mindocr.data.utils.polygon_utils import sort_clockwise


class SYNTHTEXT_Converter:
    """
    Validate polygons and sort vertices in SynthText dataset. The filtered dataset will be stored
    in the same format as the original one for compatibility purposes.

    Args:
        min_area: area below which polygons will be filtered out
    """

    def __init__(self, *args):
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
