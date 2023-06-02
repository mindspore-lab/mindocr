#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_DATA_TYPE_CONSTANT_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_DATA_TYPE_CONSTANT_H_

const int CHANNEL_SIZE = 3;

const int RGB_MAX_IMAGE_VALUE = 255;
const float RGB_MAX_IMAGE_FLOAT_VALUE = 255.0f;

const int PIX_BYTES = 4;

const int S2MS = 1000;

const int SHAPE_BATCH_SIZE_INDEX = 0;
const int SHAPE_CHANNEL_SIZE_INDEX = 1;
const int SHAPE_WIDTH_INDEX = 2;
const int SHAPE_HEIGHT_INDEX = 3;

const int MIN_DEVICE_NO = 0;
const int MAX_DEVICE_NO = 7;

const int MIN_THREAD_COUNT = 1;
const int MAX_THREAD_COUNT = 4;

const int DEFAULT_CLS_HEIGHT = 48;
const int DEFAULT_CLS_WIDTH = 192;
const float MEAN = 0.5f;

const float THRESH = 0.3;
const float BOX_THRESH = 0.5;
const int MIN_SIZE = 2;
const int MAX_SIZE = 5;
const int POINT1 = 0;
const int POINT2 = 1;
const int POINT3 = 2;
const int POINT4 = 3;
const int POINT_NUM = 4;
const int INDEX2 = 2;
const int MAX_CANDIDATES = 999;
const int MAX_VAL = 255;
const float UNCLIP_RATIO = 2.0f;
const int UNCLIP_DISTANCE = 2;

const float ROTATE_THRESH = 0.9;
#endif
