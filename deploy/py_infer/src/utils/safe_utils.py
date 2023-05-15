import contextlib
import json
import os
import re
import shutil
import stat

from .logger import logger_instance as log


def safe_list_writer(save_dict, save_path):
    """
    append the infer result to file.
    :param save_dict:
    :param save_path:
    :return:
    """
    flags, modes = os.O_WRONLY | os.O_CREAT | os.O_APPEND, stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    with os.fdopen(os.open(save_path, flags, modes), 'a') as f:
        if not save_dict:
            f.write('')
        for name, res in save_dict.items():
            content = name + '\t' + json.dumps(res, ensure_ascii=False) + "\n"
            f.write(content)


def safe_div(dividend, divisor):
    try:
        quotient = dividend / divisor
    except ZeroDivisionError as error:
        log.error(error)
        quotient = 0
    return quotient


def verify_file_size(file_path) -> bool:
    conf_file_size = os.path.getsize(file_path)
    if conf_file_size > 0 and conf_file_size / 1024 / 1024 < 10:
        return True
    return False


def valid_characters(pattern: str, characters: str) -> bool:
    if re.match(r'.*[\s]+', characters):
        return False
    if not re.match(pattern, characters):
        return False
    return True


def file_base_check(file_path: str) -> None:
    base_name = os.path.basename(file_path)
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f'the file:{base_name} does not exist!')
    if not valid_characters('^[A-Za-z0-9_+-/]+$', file_path):
        raise Exception(f'file path:{os.path.relpath(file_path)} should only include characters \'A-Za-z0-9+-_/\'!')
    if not verify_file_size(file_path):
        raise Exception(f'{base_name}: the file size must between [1, 10M]!')
    if os.path.islink(file_path):
        raise Exception(f'the file:{base_name} is link. invalid file!')
    if not os.access(file_path, mode=os.R_OK):
        raise FileNotFoundError(f'the file:{base_name} is unreadable!')


def get_safe_name(path):
    """Remove ending path separators before retrieving the basename.

    e.g. /xxx/ -> /xxx
    """
    return os.path.basename(os.path.abspath(path))


def custom_islink(path):
    """Remove ending path separators before checking soft links.

    e.g. /xxx/ -> /xxx
    """
    return os.path.islink(os.path.abspath(path))


def check_valid_dir(path):
    name = get_safe_name(path)
    check_valid_path(path, name)
    if not os.path.isdir(path):
        log.error(f'Please check if {name} is a directory.')
        raise NotADirectoryError("Check dir failed.")


def check_valid_path(path, name):
    if not path or not os.path.exists(path):
        raise FileExistsError(f'Error! {name} must exists!')
    if custom_islink(path):
        raise ValueError(f'Error! {name} cannot be a soft link!')
    if not os.access(path, mode=os.R_OK):
        raise RuntimeError(f'Error! Please check if {name} is readable.')


def check_valid_file(path, num_gb_limit=10):
    filename = get_safe_name(path)
    check_valid_path(path, filename)
    if not os.path.isfile(path):
        log.error(f'Please check if {filename} is a file.')
        raise ValueError("Check file failed.")
    check_size(path, filename, num_gb_limit=num_gb_limit)


def check_size(path, name, num_gb_limit):
    limit = num_gb_limit * 1024 * 1024 * 1024
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(f'{name} cannot be an empty file!')
    if size >= limit:
        raise ValueError(f'The size of {name} must be smaller than {num_gb_limit} GB!')


def save_path_init(path, exist_ok=False):
    if os.path.exists(path):
        if exist_ok:
            return
        shutil.rmtree(path)
    os.makedirs(path, 0o750)


@contextlib.contextmanager
def suppress_stdout():
    """
    A context manager for doing a "deep suppression" of stdout.
    """
    null_fds = os.open(os.devnull, os.O_RDWR)
    save_fds = os.dup(1)
    os.dup2(null_fds, 1)

    yield

    os.dup2(save_fds, 1)
    os.close(null_fds)
