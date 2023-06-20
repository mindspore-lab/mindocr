[English](../../en/datasets/icdar2015.md) | 中文

# 数据集下载
ICDAR 2015 [文章](https://rrc.cvc.uab.es/?ch=4)

[下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads): 在下载之前，您需要先注册一个账号。

<details>
  <summary>从何处下载 ICDAR 2015</summary>

ICDAR 2015 挑战赛分为三个任务。任务1是文本定位。任务3是单词识别。任务4是端到端文本检测识别。任务2文本分割的数据不可用。

### Text Localization

有四个与任务1相关的文件需要下载（[下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)）， 它们分别是：

```
ch4_training_images.zip
ch4_training_localization_transcription_gt.zip
ch4_test_images.zip
Challenge4_Test_Task1_GT.zip
```

### Word Recognition

有三个与任务3相关的文件需要下载（[下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)）， 它们分别是：

```
ch4_training_word_images_gt.zip
ch4_test_word_images_gt.zip
Challenge4_Test_Task3_GT.txt
```

这三个文件仅用于训练单词识别模型。训练文本检测模型不需要这三个文件。

### E2E

有九个与任务4相关的文件需要下载（[下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)）。其中包括任务1中的四个文件， 还有五个词汇文件。

```
ch4_training_vocabulary.txt
ch4_training_vocabularies_per_image.zip
ch4_test_vocabulary.txt
ch4_test_vocabularies_per_image.zip
GenericVocabulary.txt
```

如果您下载了一个名为 `Challenge4_Test_Task4_GT.zip` 的文件，请注意它与 `Challenge4_Test_Task1_GT.zip` 是相同的文件，除了名称不同。在这个repo中，我们将使用 `Challenge4_Test_Task4_GT.zip` 来代替 ICDAR2015 数据集的文件 `Challenge4_Test_Task1_GT.zip`。


</details>


在 icdar2015 下载完成以后, 请把所有的文件放在 `[path-to-data-dir]` 文件夹内，如下所示:
```
path-to-data-dir/
  ic15/
    ch4_test_images.zip
    ch4_test_vocabularies_per_image.zip
    ch4_test_vocabulary.txt
    ch4_training_images.zip
    ch4_training_localization_transcription_gt.zip
    ch4_training_vocabularies_per_image.zip
    ch4_training_vocabulary.txt
    Challenge4_Test_Task4_GT.zip
    GenericVocabulary.txt
    ch4_test_word_images_gt.zip
    ch4_training_word_images_gt.zip
    Challenge4_Test_Task3_GT.zip
```

[返回](../../../tools/dataset_converters/README_CN.md)
