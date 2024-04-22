THIRDPARTY_MODEL_PATH_DET="path/to/det/model"
THIRDPARTY_MODEL_CONFIG_DET="path/to/det/config"
THIRDPARTY_MODEL_PATH_CLS="path/to/cls/model"
THIRDPARTY_MODEL_CONFIG_CLS="path/to/cls/config"
THIRDPARTY_MODEL_PATH_REC="path/to/rec/model"
THIRDPARTY_MODEL_CONFIG_REC="path/to/rec/config"
THIRDPARTY_DICT_PATH="path/to/dict"

MINDOCR_MODEL_PATH_DET="path/to/det/model"
MINDOCR_MODEL_CONFIG_DET="path/to/det/config"
MINDOCR_MODEL_PATH_CLS="path/to/cls/config"
MINDOCR_MODEL_CONFIG_CLS="path/to/cls/config"
MINDOCR_MODEL_PATH_REC="path/to/rec/model"
MINDOCR_MODEL_CONFIG_REC="path/to/rec/config"
MINDOCR_DICT_PATH="path/to/dict"

RES_SAVE_DIR=deploy/py_infer/test/temp

### --------------- ocr_infer_server --------------------
# ocr_infer_server, det+cls+rec, thirdparty
python deploy/py_infer/example/ocr_infer_server.py \
    --det_model_path=$THIRDPARTY_MODEL_PATH_DET \
    --det_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_DET \
    --cls_model_path=$THIRDPARTY_MODEL_PATH_CLS \
    --cls_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_CLS \
    --rec_model_path=$THIRDPARTY_MODEL_PATH_REC \
    --rec_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_REC \
    --character_dict_path=$THIRDPARTY_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --result_contain_score=True \
    --node_fetch_interval=0.001 \
    --show_log=True

# ocr_infer_server, det+rec, thirdparty
python deploy/py_infer/example/ocr_infer_server.py \
    --det_model_path=$THIRDPARTY_MODEL_PATH_DET \
    --det_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_DET \
    --rec_model_path=$THIRDPARTY_MODEL_PATH_REC \
    --rec_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_REC \
    --character_dict_path=$THIRDPARTY_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --result_contain_score=True \
    --node_fetch_interval=0.001 \
    --show_log=True

# ocr_infer_server, det, thirdparty
python deploy/py_infer/example/ocr_infer_server.py \
    --det_model_path=$THIRDPARTY_MODEL_PATH_DET \
    --det_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_DET \
    --res_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# ocr_infer_server, rec, thirdparty
python deploy/py_infer/example/ocr_infer_server.py \
    --rec_model_path=$THIRDPARTY_MODEL_PATH_REC \
    --rec_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_REC \
    --character_dict_path=$THIRDPARTY_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# ocr_infer_server, cls, thirdparty
python deploy/py_infer/example/ocr_infer_server.py \
    --cls_model_path=$THIRDPARTY_MODEL_PATH_CLS \
    --cls_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_CLS \
    --res_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# ocr_infer_server, det+rec, mindocr
python deploy/py_infer/example/ocr_infer_server.py \
    --det_model_path=$MINDOCR_MODEL_PATH_DET \
    --det_model_name_or_config=$MINDOCR_MODEL_CONFIG_DET \
    --rec_model_path=$MINDOCR_MODEL_PATH_REC \
    --rec_model_name_or_config=$MINDOCR_MODEL_CONFIG_REC \
    --character_dict_path=$MINDOCR_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --result_contain_score=True \
    --node_fetch_interval=0.001 \
    --show_log=True

### --------------- infer --------------------

# infer, det+cls+rec, thirdparty
python deploy/py_infer/infer.py \
    --input_images_dir=deploy/py_infer/example/dataset/det \
    --det_model_path=$THIRDPARTY_MODEL_PATH_DET \
    --det_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_DET \
    --rec_model_path=$THIRDPARTY_MODEL_PATH_REC \
    --rec_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_REC \
    --cls_model_path=$THIRDPARTY_MODEL_PATH_CLS \
    --cls_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_CLS \
    --character_dict_path=$THIRDPARTY_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --vis_pipeline_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# infer, det+rec, thirdparty
python deploy/py_infer/infer.py \
    --input_images_dir=deploy/py_infer/example/dataset/det \
    --det_model_path=$THIRDPARTY_MODEL_PATH_DET \
    --det_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_DET \
    --rec_model_path=$THIRDPARTY_MODEL_PATH_REC \
    --rec_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_REC \
    --character_dict_path=$THIRDPARTY_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --vis_pipeline_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# infer, det, thirdparty
python deploy/py_infer/infer.py \
    --input_images_dir=deploy/py_infer/example/dataset/det \
    --det_model_path=$THIRDPARTY_MODEL_PATH_DET \
    --det_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_DET \
    --res_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# infer, rec, thirdparty
python deploy/py_infer/infer.py \
    --input_images_dir=deploy/py_infer/example/dataset/cls_rec \
    --rec_model_path=$THIRDPARTY_MODEL_PATH_REC \
    --rec_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_REC \
    --character_dict_path=$THIRDPARTY_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# infer, cls, thirdparty
python deploy/py_infer/infer.py \
    --input_images_dir=deploy/py_infer/example/dataset/cls_rec \
    --cls_model_path=$THIRDPARTY_MODEL_PATH_CLS \
    --cls_model_name_or_config=$THIRDPARTY_MODEL_CONFIG_CLS \
    --res_save_dir=$RES_SAVE_DIR \
    --show_log=True \
    --result_contain_score=False \
    --node_fetch_interval=0.001

# infer, det+cls+rec, mindocr
python deploy/py_infer/infer.py \
    --input_images_dir=deploy/py_infer/example/dataset/det \
    --det_model_path=$MINDOCR_MODEL_PATH_DET \
    --det_model_name_or_config=$MINDOCR_MODEL_CONFIG_DET \
    --rec_model_path=$MINDOCR_MODEL_PATH_REC \
    --rec_model_name_or_config=$MINDOCR_MODEL_CONFIG_REC \
    --character_dict_path=$MINDOCR_DICT_PATH \
    --res_save_dir=$RES_SAVE_DIR \
    --result_contain_score=True \
    --node_fetch_interval=0.001 \
    --show_log=True
