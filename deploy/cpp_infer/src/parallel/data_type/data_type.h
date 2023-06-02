#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_DATA_TYPE_DATA_TYPE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_DATA_TYPE_DATA_TYPE_H_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"
#include "include/api/model.h"

// Class for resized image info
class __attribute__((visibility("default"))) ResizedImageInfo {
 public:
  uint32_t widthResize;    // memoryWidth
  uint32_t heightResize;   // memoryHeight
  uint32_t widthOriginal;  // imageWidth
  uint32_t heightOriginal;  // imageHeight
  float ratio;
};

// Class for text object detection
class __attribute__((visibility("default"))) TextObjectInfo {
 public:
  float x0;
  float y0;
  float x1;
  float y1;
  float x2;
  float y2;
  float x3;
  float y3;
  float confidence;
  std::string result;
};

// Class for text generation (i.e. translation, OCR)
class __attribute__((visibility("default"))) TextsInfo {
 public:
  std::vector<std::string> text;
};

enum class BackendType {
  LITE = 0,
  ACL = 1,
  UNSUPPORTED = -1
};

enum class TaskType {
  DET = 0,
  CLS = 1,
  REC = 2,
  DET_REC = 3,
  DET_CLS_REC = 4,
  UNSUPPORTED = 5
};

struct CommonData {
  // modules info
  int channelId = {};
  int imgId = {};
  int imgTotal = {};
  int subImgTotal = {};

  // img info
  std::string imgPath = {};
  std::string imgName = {};
  uint32_t srcWidth = {};
  uint32_t srcHeight = {};
  cv::Mat frame = {};
  std::vector<cv::Mat> imgMatVec = {};

  // infer related
  uint8_t *imgBuffer = {};
  std::vector<MxBase::Tensor> outputMindXTensorVec = {};
  std::vector<mindspore::MSTensor> outputLiteTensorVec = {};
  std::vector<ResizedImageInfo> resizedImageInfos = {};

  // det info
  uint32_t resizeWidth = {};
  uint32_t resizeHeight = {};
  float ratio = {};

  // cls and rec
  uint32_t batchSize = {};
  uint32_t frameSize = {};
  bool eof = {};

  // rec
  int maxResizedW = {};
  float maxWHRatio = {};

  // infer res
  std::string imageName = {};
  std::vector<std::string> inferRes = {};

  BackendType backendType = BackendType::LITE;
};

struct LiteModelWrap {
  mindspore::Model *model;
  std::vector<std::vector<uint64_t>> dynamicGearInfo = {};
};

#endif
