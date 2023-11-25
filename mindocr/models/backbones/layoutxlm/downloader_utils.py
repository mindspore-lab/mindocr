import hashlib
import logging
import os
import os.path as osp
import shutil
import tarfile
import zipfile
from typing import Optional

import requests
from tqdm.auto import tqdm

logger = logging.getLogger("mindocr.downloader")

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith("http://") or path.startswith("https://")


def url_file_exists(url: str) -> bool:
    """check whether the url file exists

        refer to: https://stackoverflow.com/questions/2486145/python-check-if-url-to-jpg-exists

    Args:
        url (str): the url of target file

    Returns:
        bool: whether the url file exists
    """
    if not is_url(url):
        return False

    result = requests.head(url)
    return result.status_code == requests.codes.ok


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info("File {} md5 check failed, {}(calc) != " "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    os.makedirs(path, exist_ok=True)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. " "Retry limit reached".format(url))

        logger.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code " "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size:
                with tqdm(total=int(total_size), unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if "/" in file_path:
            file_path = file_path.replace("/", os.sep)
        elif "\\" in file_path:
            file_path = file_path.replace("\\", os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
        return True
    return False


def _uncompress_file_tar(filepath, mode="r:*"):
    files = tarfile.open(filepath, mode)
    file_list = files.getnames()
    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
        files.extractall(file_dir, files.getmembers())
    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        files.extractall(file_dir, files.getmembers())
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)

        files.extractall(os.path.join(file_dir, rootpath), files.getmembers())

    files.close()

    return uncompressed_path


def _uncompress_file_zip(filepath):
    files = zipfile.ZipFile(filepath, "r")
    file_list = files.namelist()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)
        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    return uncompressed_path


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True):
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    assert is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)

    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info("Found {}".format(fullpath))
    else:
        fullpath = _download(url, root_dir, md5sum)

    if tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath):
        fullpath = _decompress(fullpath)

    # model tokenizer config, [file-lock]
    return fullpath


def get_path_from_url_with_filelock(
        url: str, root_dir: str, md5sum: Optional[str] = None, check_exist: bool = True, timeout: float = -1
) -> str:
    """construct `get_path_from_url` for `model_utils` to enable downloading multiprocess-safe

    Args:
        url (str): the url of resource file
        root_dir (str): the local download path
        md5sum (str, optional): md5sum string for file. Defaults to None.
        check_exist (bool, optional): whether check the file is exist. Defaults to True.
        timeout (int, optional): the timeout for downloading. Defaults to -1.

    Returns:
        str: the path of downloaded file
    """
    from filelock import FileLock
    os.makedirs(root_dir, exist_ok=True)

    # create lock file, which is empty, under the `LOCK_FILE_HOME` directory.
    lock_file_name = hashlib.md5((url + root_dir).encode("utf-8")).hexdigest()

    # create `.lock` private directory in the cache dir
    lock_file_path = os.path.join(root_dir, ".lock", lock_file_name)

    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

    with FileLock(lock_file_path, timeout=timeout):
        result = get_path_from_url(url=url, root_dir=root_dir, md5sum=md5sum, check_exist=check_exist)
    return result
