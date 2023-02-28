Convert ocr annotation to the required format

## Text Detection/Spotting Annotation

The format of the converted annotation file should follw:
``` text
ch4_test_images/img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]   
``` 

To convert, please run `python tools/dataset_converters/convert.py` 


## Text Recognition Annotation

The annotation format for text recognition dataset follows 
```text 
word_7.png	fusionopolis
word_8.png	fusionopolis
word_9.png	Reserve
word_10.png	CAUTION
word_11.png	citi
```

Image path and text label are seperated by \t. 


To convert, please run:
``` python
python tools/dataset_converters/convert.py --mode=rec --input_path /data/ocr_datasets/ic15/word_recognition/ch4_training_word_images_gt/gt.txt --output_label /data/ocr_datasets/ic15/word_recognition/rec_gt_train.txt
```

