import sys
import os
import subprocess

python_path = sys.executable

ori_path = os.getcwd()
curr_file_dir = os.path.split(os.path.realpath(__file__))[0]  # get current file dir
os.chdir(curr_file_dir)
files = os.listdir()
if 'pse.cpp' not in files or 'build' not in files:  # check whether pse codes have been compiled
    if subprocess.call(
            '{} setup.py build_ext --inplace'.format(python_path), shell=True) != 0:
        raise RuntimeError(
            'Cannot compile pse: {}, please check whether Cython is installed and '
            'refer to the compilation guide(/mindocr/postprocess/pse/README.md)'.
            format(os.path.dirname(os.path.realpath(__file__))))
os.chdir(ori_path)

from .pse import pse
