"""
Script to combine lmdb datasets into single one
"""
import argparse
import os
import sys
from typing import Any, Dict, Generator, List, Tuple

import lmdb

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from tools.dataset_converters.utils.lmdb_writer import create_lmdb_dataset


def load_hierarchical_lmdb_dataset(root: str) -> List[Dict[str, Any]]:
    lmdbs_info = list()
    for rootdir, dirs, _ in os.walk(root):
        if not dirs:
            try:
                env = lmdb.Environment(
                    rootdir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False
                )
            except lmdb.Error:
                continue
            txn = env.begin(write=False)
            data_size = int(txn.get("num-samples".encode()))
            lmdb_info = {"rootdir": rootdir, "env": env, "txn": txn, "data_size": data_size}
            lmdbs_info.append(lmdb_info)
    return lmdbs_info


def get_lmdb_sample_info(txn: lmdb.Transaction, idx: int) -> Tuple[bytes, bytes]:
    label_key = "label-%09d".encode() % idx
    label = txn.get(label_key).decode("utf-8")
    img_key = "image-%09d".encode() % idx
    imgbuf = txn.get(img_key)
    return imgbuf, label


def yield_combined_record(root: str) -> Generator[Tuple[bytes, bytes], Any, None]:
    lmdbs_info = load_hierarchical_lmdb_dataset(root)
    for lmdb_info in lmdbs_info:
        for idx in range(lmdb_info["data_size"]):
            yield get_lmdb_sample_info(lmdb_info["txn"], idx + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple LMDB record into the single one.")
    parser.add_argument("root", help="Path of the root directory")
    parser.add_argument(
        "-o",
        "--output_path",
        default="./lmdb_out",
        help="Path to save the combined LMDB file.",
    )
    args = parser.parse_args()
    create_lmdb_dataset(records=yield_combined_record(args.root), output_path=args.output_path)
