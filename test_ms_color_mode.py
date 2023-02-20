#!/usr/bin/env python
# coding=utf8
'''*************************************************************************
    > File Name: test_ms_color_mode.py
    > Author: HUANG Yongxiang
    > Mail:
    > Created Time: Mon 13 Feb 23:17:28 2023
    > Usage:
*************************************************************************'''

import mindspore as ms
import mindcv

#dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, download=True, shuffle=False)
ds = mindcv.create_dataset('cifar10')
print(ds)
