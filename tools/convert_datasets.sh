#!/bin/bash
DATASETS_DIR="../ocr_datasets" # the directory containing multiple ocr datasets that have been downloaded and correctly placed as:
# DATASETS_DIR/
#   ic15/
#     ch4_test_images.zip
#     ch4_test_vocabularies_per_image.zip
#     ch4_test_vocabulary.txt
#     ch4_training_images.zip
#     ch4_training_localization_transcription_gt.zip
#     ch4_training_vocabularies_per_image.zip
#     ch4_training_vocabulary.txt
#     Challenge4_Test_Task4_GT.zip
#     GenericVocabulary.txt
#     ch4_test_word_images_gt.zip
#     ch4_training_word_images_gt.zip
#     Challenge4_Test_Task3_GT.zip
#   mlt2017/
#     ch8_training_images_1.zip
#     ch8_training_images_2.zip
#     ch8_training_images_3.zip
#     ch8_training_images_4.zip
#     ch8_training_images_5.zip
#     ch8_training_images_6.zip
#     ch8_training_images_7.zip
#     ch8_training_images_8.zip
#     ch8_training_localization_transcription_gt_v2.zip
#     ch8_validation_images.zip
#     ch8_validation_localization_transcription_gt_v2.zip
#     ch8_training_word_images_gt_part_1.zip
#     ch8_training_word_images_gt_part_2.zip
#     ch8_training_word_images_gt_part_3.zip
#     ch8_training_word_gt_v2.zip
#     ch8_validation_word_images_gt.zip
#     ch8_validation_word_gt_v2.zip
#   syntext150k/
#     syntext1/
#       images.zip
#       annotations/
#         ecms_v1_maxlen25.json
#     syntext2/
#       images.zip
#       annotations/
#         syntext_word_eng.json
#   totaltext/
#     totaltext.zip
#     txt_format.zip
# The information about how to download and place these datasets can be found in docs/datasets 
#########################icdar2015#########################
DIR="$DATASETS_DIR/ic15"

if  [ ! -d $DIR ] || [  ! "$(ls -A $DIR)"  ]; then
  echo "ICDAR 15 is Empty! Skipped."
else
  unzip $DIR/ch4_training_images.zip -d $DIR/ch4_training_images/
  rm $DIR/ch4_training_images.zip

  unzip $DIR/ch4_training_localization_transcription_gt.zip -d $DIR/ch4_training_localization_transcription_gt/
  rm $DIR/ch4_training_localization_transcription_gt.zip

  unzip $DIR/ch4_training_vocabularies_per_image.zip -d $DIR/ch4_training_vocabularies_per_image/
  rm $DIR/ch4_training_vocabularies_per_image.zip

  unzip $DIR/ch4_test_images.zip -d $DIR/ch4_test_images/
  rm $DIR/ch4_test_images.zip

  unzip $DIR/ch4_test_vocabularies_per_image.zip -d $DIR/ch4_test_vocabularies_per_image/
  rm $DIR/ch4_test_vocabularies_per_image.zip


  unzip $DIR/Challenge4_Test_Task4_GT.zip -d $DIR/Challenge4_Test_Task4_GT/
  rm $DIR/Challenge4_Test_Task4_GT.zip

  unzip $DIR/ch4_training_word_images_gt.zip -d $DIR/ch4_training_word_images_gt/
  rm $DIR/ch4_training_word_images_gt.zip

  unzip $DIR/ch4_test_word_images_gt.zip -d $DIR/ch4_test_word_images_gt/
  rm $DIR/ch4_test_word_images_gt.zip

  if test -f "$DIR/train_det_gt.txt"; then
     echo "$DIR/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  ic15 \
          --task det \
          --image_dir $DIR/ch4_training_images/ \
          --label_dir $DIR/ch4_training_localization_transcription_gt/ \
          --output_path $DIR/train_det_gt.txt
  fi
  if test -f "$DIR/test_det_gt.txt"; then
     echo "$DIR/test_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  ic15 \
          --task det \
          --image_dir $DIR/ch4_test_images/ \
          --label_dir $DIR/Challenge4_Test_Task4_GT/ \
          --output_path $DIR/test_det_gt.txt
  fi
  if test -f "$DIR/train_rec_gt.txt"; then
     echo "$DIR/train_rec_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task rec \
        --label_dir $DIR/ch4_training_word_images_gt/gt.txt \
        --output_path $DIR/train_rec_gt.txt
  fi
  if test -f "$DIR/test_rec_gt.txt"; then
     echo "$DIR/test_rec_gt.txt exists."
  else
    python tools/dataset_converters/convert.py \
        --dataset_name  ic15 \
        --task rec \
        --label_dir $DIR/Challenge4_Test_Task3_GT.txt \
        --output_path $DIR/test_rec_gt.txt
  fi
fi

##########################syntext150k#########################
DIR="$DATASETS_DIR/syntext150k"

if  [ ! -d $DIR ] || [  ! "$(ls -A $DIR)"  ]; then
  echo "SynText150k is Empty! Skipped."
else

  unzip $DIR/syntext1/images.zip -d $DIR/syntext1/images
  rm $DIR/syntext1/images.zip

  unzip $DIR/syntext2/images.zip -d $DIR/syntext2/images
  rm $DIR/syntext2/images.zip

  if test -f "$DIR/syntext1/train_det_gt.txt"; then
     echo "$DIR/syntext1/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  syntext150k \
          --task det \
          --image_dir $DIR/syntext1/images/emcs_imgs/ \
          --label_dir $DIR/syntext1/annotations/ecms_v1_maxlen25.json \
          --output_path $DIR/syntext1/train_det_gt.txt
  fi

  if test -f "$DIR/syntext2/train_det_gt.txt"; then
     echo "$DIR/syntext2/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  syntext150k \
          --task det \
          --image_dir $DIR/syntext2/images/syntext_word_eng/ \
          --label_dir $DIR/syntext2/annotations/syntext_word_eng.json \
          --output_path $DIR/syntext2/train_det_gt.txt
  fi
fi

##########################mlt2017#########################
DIR="$DATASETS_DIR/mlt2017"
if  [ ! -d $DIR ] || [  ! "$(ls -A $DIR)"  ]; then
   echo "MLT2017 is Empty! Skipped."
else
   unzip $DIR/ch8_training_images_1.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_1.zip
   unzip $DIR/ch8_training_images_2.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_2.zip
   unzip $DIR/ch8_training_images_3.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_3.zip
   unzip $DIR/ch8_training_images_4.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_4.zip
   unzip $DIR/ch8_training_images_5.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_5.zip
   unzip $DIR/ch8_training_images_6.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_6.zip
   unzip $DIR/ch8_training_images_7.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_7.zip
   unzip $DIR/ch8_training_images_8.zip -d $DIR/ch8_training_images/
   rm $DIR/ch8_training_images_8.zip

   unzip $DIR/ch8_training_localization_transcription_gt_v2.zip -d $DIR/ch8_training_localization_transcription_gt/
   rm $DIR/ch8_training_localization_transcription_gt_v2.zip

   unzip $DIR/ch8_validation_images.zip -d $DIR/ch8_validation_images/
   rm $DIR/ch8_validation_images.zip

   unzip $DIR/ch8_validation_localization_transcription_gt_v2.zip -d $DIR/ch8_validation_localization_transcription_gt/
   rm $DIR/ch8_validation_localization_transcription_gt_v2.zip

   unzip $DIR/ch8_training_word_images_gt_part_1.zip -d $DIR/ch8_training_word_images_gt/
   rm $DIR/ch8_training_word_images_gt_part_1.zip
   unzip $DIR/ch8_training_word_images_gt_part_2.zip -d $DIR/ch8_training_word_images_gt/
   rm $DIR/ch8_training_word_images_gt_part_2.zip
   unzip $DIR/ch8_training_word_images_gt_part_3.zip -d $DIR/ch8_training_word_images_gt/
   rm $DIR/ch8_training_word_images_gt_part_3.zip

   unzip $DIR/ch8_training_word_gt_v2.zip -d $DIR/ch8_training_word_gt/
   rm $DIR/ch8_training_word_gt_v2.zip
   unzip $DIR/ch8_validation_word_images_gt.zip -d $DIR/ch8_validation_word_images_gt/
   rm $DIR/ch8_validation_word_images_gt.zip
   unzip $DIR/ch8_validation_word_gt_v2.zip -d $DIR/ch8_validation_word_gt/
   rm $DIR/ch8_validation_word_gt_v2.zip
   
  if test -f "$DIR/train_det_gt.txt"; then
     echo "$DIR/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  mlt2017 \
          --task det \
          --image_dir $DIR/ch8_training_images/ \
          --label_dir $DIR/ch8_training_localization_transcription_gt/\
          --output_path $DIR/train_det_gt.txt
  fi
  if test -f "$DIR/val_det_gt.txt"; then
     echo "$DIR/val_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  mlt2017 \
          --task det \
          --image_dir $DIR/ch8_validation_images/ \
          --label_dir $DIR/ch8_validation_localization_transcription_gt/ \
          --output_path $DIR/val_det_gt.txt
  fi
 

  if test -f "$DIR/train_rec_gt.txt"; then
     echo "$DIR/train_rec_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
        --dataset_name  mlt2017 \
        --task rec \
        --label_dir $DIR/ch8_training_word_gt/gt.txt \
        --output_path $DIR/train_rec_gt.txt
  fi
  if test -f "$DIR/val_rec_gt.txt"; then
     echo "$DIR/val_rec_gt.txt exists."
  else
    python tools/dataset_converters/convert.py \
        --dataset_name  mlt2017 \
        --task rec \
        --label_dir $DIR/ch8_validation_word_gt/gt.txt \
        --output_path $DIR/val_rec_gt.txt
  fi
fi
##########################total_text#########################
DIR="$DATASETS_DIR/totaltext"
if  [ ! -d $DIR ] || [  ! "$(ls -A $DIR)"  ]; then
  echo "Total-Text is Empty! Skipped."
else
  unzip $DIR/totaltext.zip -d  $DIR/
  rm $DIR/totaltext.zip

  unzip $DIR/txt_format.zip -d  $DIR/annotations
  rm $DIR/txt_format.zip 


  if test -f "$DIR/train_det_gt.txt"; then
     echo "$DIR/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  totaltext \
          --task det \
          --image_dir $DIR/Images/Train/ \
          --label_dir $DIR/annotations/Train/ \
          --output_path $DIR/train_det_gt.txt
  fi
  if test -f "$DIR/test_det_gt.txt"; then
     echo "$DIR/test_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  totaltext \
          --task det \
          --image_dir $DIR/Images/Test/ \
          --label_dir $DIR/annotations/Test/ \
          --output_path $DIR/test_det_gt.txt
  fi
fi
##########################td500#########################
DIR="$DATASETS_DIR/td500"
if  [ ! -d $DIR ] || [  ! "$(ls -A $DIR)"  ]; then
  echo "td500 is Empty! Skipped."
else
  unzip $DIR/MSRA-TD500.zip -d  $DIR/
  rm $DIR/MSRA-TD500.zip


  if test -f "$DIR/MSRA-TD500/train_det_gt.txt"; then
     echo "$DIR/MSRA-TD500/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  td500 \
          --task det \
          --image_dir $DIR/MSRA-TD500/train/ \
          --label_dir $DIR/MSRA-TD500/train/ \
          --output_path $DIR/MSRA-TD500/train_det_gt.txt
  fi
  if test -f "$DIR/MSRA-TD500/test_det_gt.txt"; then
     echo "$DIR/MSRA-TD500/test_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  td500 \
          --task det \
          --image_dir $DIR/MSRA-TD500/test/ \
          --label_dir $DIR/MSRA-TD500/test/ \
          --output_path $DIR/MSRA-TD500/test_det_gt.txt
  fi
fi

##########################ctw1500#########################
DIR="$DATASETS_DIR/ctw1500"
if  [ ! -d $DIR ] || [  ! "$(ls -A $DIR)"  ]; then
  echo "ctw1500 is Empty! Skipped."
else
  unzip $DIR/train_images.zip -d  $DIR/
  rm $DIR/train_images.zip

  unzip $DIR/test_images.zip -d  $DIR/
  rm $DIR/test_images.zip

  unzip $DIR/ctw1500_train_labels.zip -d  $DIR/
  rm $DIR/ctw1500_train_labels.zip

  unzip $DIR/gt_ctw1500.zip -d  $DIR/gt_ctw1500/
  rm $DIR/gt_ctw1500.zip

  if test -f "$DIR/train_det_gt.txt"; then
     echo "$DIR/train_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  ctw1500 \
          --task det \
          --image_dir $DIR/train_images/ \
          --label_dir $DIR/ctw1500_train_labels/ \
          --output_path $DIR/train_det_gt.txt
  fi
  if test -f "$DIR/test_det_gt.txt"; then
     echo "$DIR/test_det_gt.txt exists."
  else
     python tools/dataset_converters/convert.py \
          --dataset_name  ctw1500 \
          --task det \
          --image_dir $DIR/test_images/ \
          --label_dir $DIR/gt_ctw1500/ \
          --output_path $DIR/test_det_gt.txt
  fi
fi
