# 数据下载

SynthText是一个合成生成的数据集，其中单词实例被放置在自然场景图像中，并考虑了场景布局。

[论文](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/) | [下载SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)

下载`SynthText.zip`文件并解压缩到`[path-to-data-dir]`文件夹中：
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

> :warning: 另外, 我们强烈建议在使用 `SynthText` 数据集之前先进行预处理，因为它包含一些错误的数据。可以使用下列的方式进行校正:
> ```shell
> python tools/dataset_converters/convert.py --dataset_name=synthtext --task=det --label_dir=/path-to-data-dir/SynthText/gt.mat --output_path=/path-to-data-dir/SynthText/gt_processed.mat
> ```
> 以上的操作会产生与`SynthText`原始标注格式相同但是是经过过滤后的标注数据.


[返回dataset converters](converters.md)
