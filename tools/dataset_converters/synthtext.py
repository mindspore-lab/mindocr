from typing import Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, savemat
from shapely.geometry import Polygon

from mindocr.data.utils.polygon_utils import sort_clockwise


class SYNTHTEXT_Converter:
    """
    Validate polygons and sort vertices in SynthText dataset. The filtered dataset will be stored
    in the same format as the original one for compatibility purposes.
    """
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _sort_and_validate(sample: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
        """
        Sort vertices in clockwise order (to eliminate self-intersection) and filter invalid polygons out.
        Args:
            sample: tuple containing polygons and texts instances.
        Returns:
            filtered polygons and texts.
        """
        polys, texts = sample
        polys = polys.transpose().reshape(-1, 4, 2)     # some labels have (4, 2) shape (no batch dimension)
        texts = [t for text in texts.tolist() for t in text.split()]    # TODO: check the correctness of texts order

        # Sort vertices clockwise and filter invalid polygons out
        new_polys, new_texts = [], []
        for poly, text in zip(polys, texts):
            poly = sort_clockwise(poly)
            if Polygon(poly).is_valid:
                new_polys.append(poly)
                new_texts.append(text)

        return np.array(new_polys).transpose(), np.array(new_texts)     # preserve polygons' axes order

    def convert(self, task='det', image_dir=None, label_path=None, output_path=None):
        print('Loading SynthText dataset. It might take a while...')
        mat = loadmat(label_path)

        # use multiprocessing to process the dataset faster
        with ProcessPoolExecutor(max_workers=8) as pool:
            data_list = list(tqdm(pool.map(self._sort_and_validate, zip(mat['wordBB'][0], mat['txt'][0])),
                                  total=len(mat['imnames'][0]), desc='Processing data', miniters=10000))

        wordBB, txt = zip(*data_list)
        for i in range(len(mat['wordBB'][0])):  # how to stack wordBB?
            mat['wordBB'][0][i] = wordBB[i]
        mat['txt'] = np.array(txt).reshape(1, -1)

        print('Saving...')
        savemat(output_path,
                {'charBB': mat['charBB'],   # save as it is
                 'wordBB': mat['wordBB'],
                 'imnames': mat['imnames'],
                 'txt': mat['txt']},
                do_compression=True)
