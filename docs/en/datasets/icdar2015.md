# Data Downloading
ICDAR 2015 [paper](https://rrc.cvc.uab.es/?ch=4)

[download source](https://rrc.cvc.uab.es/?ch=4&com=downloads): one must register an account to download the dataset.

<details>
  <summary>Where to Download ICDAR 2015</summary>

ICDAR 2015 Challenge has three tasks. Task 1 is Text Localization. Task 3 is Word Recognition. Task 4 is End-to-end Text Spotting. Task 2 Text Segmentation is not available.

### Text Localization

The four files downloaded from [web](https://rrc.cvc.uab.es/?ch=4&com=downloads) for task 1 are
```
ch4_training_images.zip
ch4_training_localization_transcription_gt.zip
ch4_test_images.zip
Challenge4_Test_Task1_GT.zip
```

### Word Recognition

The three files downloaded from [web](https://rrc.cvc.uab.es/?ch=4&com=downloads) for task 3 are
```
ch4_training_word_images_gt.zip
ch4_test_word_images_gt.zip
Challenge4_Test_Task3_GT.txt
```
The three files are only needed for training word recognition models. Training text detection models does not require the three files.





### E2E

The nine files downloaded from [web](https://rrc.cvc.uab.es/?ch=4&com=downloads) for task 4 are the union of the four files in the text localization task (task 1) and five vocabulary files
```
ch4_training_vocabulary.txt
ch4_training_vocabularies_per_image.zip
ch4_test_vocabulary.txt
ch4_test_vocabularies_per_image.zip
GenericVocabulary.txt
```
If you download a file named `Challenge4_Test_Task4_GT.zip`, please note that it is the same file as `Challenge4_Test_Task1_GT.zip`, except for its name. In this repository, we will use `Challenge4_Test_Task4_GT.zip` for ICDAR2015 dataset.

</details>


After downloading the icdar2015 dataset, place all the files under `[path-to-data-dir]` folder:
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

[Back to dataset converters](converters.md)
