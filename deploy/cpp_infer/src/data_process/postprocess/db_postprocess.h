#ifndef DEPLOY_CPP_INFER_SRC_DATA_PROCESS_POSTPROCESS_DB_POSTPROCESS_H_
#define DEPLOY_CPP_INFER_SRC_DATA_PROCESS_POSTPROCESS_DB_POSTPROCESS_H_

#include <vector>
#include <opencv2/imgproc.hpp>
#include "Log/Log.h"
#include "utils/utils.h"
#include "data_type/data_type.h"
#include "data_type/constant.h"
#include "status_code/status_code.h"

class DBPostprocess {
 public:
  DBPostprocess();

  ~DBPostprocess() = default;

  Status DbnetMindXObjectDetectionOutput(const std::vector<MxBase::Tensor> &singleResult,
                                         std::vector<std::vector<TextObjectInfo>> *textObjInfos,
                                         const std::vector<ResizedImageInfo> &resizedImageInfos);

  Status DbnetLiteObjectDetectionOutput(const std::vector<mindspore::MSTensor> &singleResult,
                                        std::vector<std::vector<TextObjectInfo>> *textObjInfos,
                                        const std::vector<ResizedImageInfo> &resizedImageInfos);

 private:
  void FilterByMinSize(const std::vector<cv::Point> &contour, std::vector<cv::Point2f> *box, float *minSide);

  void FilterByBoxScore(const cv::Mat &prediction, const std::vector<cv::Point2f> &box, float *score);

  void FilterByClippedMinSize(std::vector<cv::Point2f> *box, float *minSide);

  void ConstructInfo(std::vector<TextObjectInfo> *textObjectInfo, const std::vector<cv::Point2f> &box,
                     const std::vector<ResizedImageInfo> &resizedImageInfos, const uint32_t &index, float score);

  int NpClip(const int &coordinate, const int &sideLen);

  float PointsL2Distance(cv::Point2f p1, cv::Point2f p2);

  static bool SortByX(cv::Point2f p1, cv::Point2f p2);

  static bool SortByY(cv::Point2f p1, cv::Point2f p2);

  int minSize_ = MIN_SIZE;
  float thresh_ = THRESH;
  float boxThresh_ = BOX_THRESH;
  uint32_t resizedW_{};
  uint32_t resizedH_{};

  float unclipRatio_ = UNCLIP_RATIO;
  int candidates_ = MAX_CANDIDATES;

  void FindContours(std::vector<std::vector<TextObjectInfo>> *textObjInfos,
                    const std::vector<ResizedImageInfo> &resizedImageInfos, uint32_t i, const std::vector<uchar> &prob,
                    const std::vector<float> &fprob);
};

#endif
