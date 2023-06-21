# Data Downloading

SynthText is a synthetically generated dataset, in which word instances are placed in natural scene images, while taking into account the scene layout.
[Paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/) | [Download SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)


Download the `SynthText.zip` file and unzip in `[path-to-data-dir]` folder:
```
path-to-data-dir/
 ├── SynthText/
 │   ├── 1/
 │   │   ├── ant+hill_1_0.jpg
 │   │   └── ...
 │   ├── 2/
 │   │   ├── ant+hill_4_0.jpg
 │   │   └── ...
 │   ├── ...
 │   └── gt.mat
```

> :warning: Additionally, It is strongly recommended to pre-process the `SynthText` dataset before using it as it contains some faulty data:
> ```shell
> python tools/dataset_converters/convert.py --dataset_name=synthtext --task=det --label_dir=/path-to-data-dir/SynthText/gt.mat --output_path=/path-to-data-dir/SynthText/gt_processed.mat
> ```
> This operation will generate a filtered output in the same format as the original `SynthText`.

[Back to dataset converters](converters.md)
