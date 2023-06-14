#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "postprocess/clipper.hpp"
#include "postprocess/db_postprocess.h"

DBPostprocess::DBPostprocess() = default;

Status DBPostprocess::DbnetMindXObjectDetectionOutput(const std::vector<MxBase::Tensor> &singleResult,
                                                  std::vector<std::vector<TextObjectInfo>> *textObjInfos,
                                                  const std::vector<ResizedImageInfo> &resizedImageInfos) {
  LogDebug << "DbnetPost start to write results.";
  uint32_t batchSize = singleResult.size();
  for (uint32_t i = 0; i < batchSize; i++) {
    auto probMap = reinterpret_cast<float *>(singleResult[0].GetData());
    this->resizedH_ = resizedImageInfos[i].heightResize;
    this->resizedW_ = resizedImageInfos[i].widthResize;
    std::vector<uchar> prob(resizedW_ * resizedH_, ' ');
    std::vector<float> fprob(resizedW_ * resizedH_, 0.f);
    for (size_t j = 0; j < resizedW_ * resizedH_; ++j) {
      prob[j] = (uchar)(probMap[j] * MAX_VAL);
      fprob[j] = static_cast<float>(probMap[j]);
    }
    FindContours(textObjInfos, resizedImageInfos, i, prob, fprob);
  }
  return Status::OK;
}

Status DBPostprocess::DbnetLiteObjectDetectionOutput(const std::vector<mindspore::MSTensor> &singleResult,
                                                 std::vector<std::vector<TextObjectInfo>> *textObjInfos,
                                                 const std::vector<ResizedImageInfo> &resizedImageInfos) {
  LogDebug << "DbnetPost start to write results.";
  uint32_t batchSize = singleResult.size();
  for (uint32_t i = 0; i < batchSize; i++) {
    auto probMap = reinterpret_cast<const float *>((singleResult[0].Data().get()));
    this->resizedH_ = resizedImageInfos[i].heightResize;
    this->resizedW_ = resizedImageInfos[i].widthResize;
    std::vector<uchar> prob(resizedW_ * resizedH_, ' ');
    std::vector<float> fprob(resizedW_ * resizedH_, 0.f);
    for (size_t j = 0; j < resizedW_ * resizedH_; ++j) {
      prob[j] = (uchar)(probMap[j] * MAX_VAL);
      fprob[j] = static_cast<float>(probMap[j]);
    }
    FindContours(textObjInfos, resizedImageInfos, i, prob, fprob);
  }
  return Status::OK;
}

void DBPostprocess::FindContours(std::vector<std::vector<TextObjectInfo>> *textObjInfos,
                             const std::vector<ResizedImageInfo> &resizedImageInfos, uint32_t i,
                             const std::vector<uchar> &prob, const std::vector<float> &fprob) {
  cv::Mat mask(resizedH_, resizedW_, CV_8UC1, const_cast<uchar *>(static_cast<const uchar *>(prob.data())));
  cv::Mat prediction(resizedH_, resizedW_, CV_32F, const_cast<float *>(static_cast<const float *>(fprob.data())));
  cv::Mat binmask;

  cv::threshold(mask, binmask, static_cast<float>((thresh_ * MAX_VAL)), MAX_VAL, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binmask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  int contourNum = NpClip(contours.size(), candidates_);
  std::vector<TextObjectInfo> textObjectInfo;
  // traverse and filter all contours
  for (int j = 0; j < contourNum; j++) {
    std::vector<cv::Point> contour = contours[j];
    std::vector<cv::Point2f> box;
    float minSide1 = 0.f;
    float minSide2 = 0.f;
    float score = 0.f;
    // 1st filter
    FilterByMinSize(contour, &box, &minSide1);
    if (minSide1 < minSize_) {
      continue;
    }
    // 2nd filter
    FilterByBoxScore(prediction, box, &score);
    if (score < boxThresh_) {
      continue;
    }
    // 3rd filter
    FilterByClippedMinSize(&box, &minSide2);
    if (minSide2 < minSize_ + UNCLIP_DISTANCE) {
      continue;
    }
    // write box info into TextObjectInfo
    ConstructInfo(&textObjectInfo, box, resizedImageInfos, i, score);
  }
  textObjInfos->emplace_back(std::move(textObjectInfo));
}

void DBPostprocess::ConstructInfo(std::vector<TextObjectInfo> *textObjectInfo, const std::vector<cv::Point2f> &box,
                              const std::vector<ResizedImageInfo> &resizedImageInfos, const uint32_t &index,
                              float score) {
  if (resizedImageInfos.size() < index) {
    LogError << "ResizedImageInfos is empty";
    return;
  }
  uint32_t originWidth = resizedImageInfos[index].widthOriginal;
  uint32_t originHeight = resizedImageInfos[index].heightOriginal;
  if (originWidth == 0 || originHeight == 0) {
    LogError << GetError(APP_ERR_DIVIDE_ZERO) << "the origin width or height must not equal to 0!";
    return;
  }
  if (resizedW_ == 0 || resizedH_ == 0) {
    LogError << GetError(APP_ERR_DIVIDE_ZERO) << "the resized width or height must not equal to 0!";
    return;
  }
  float ratio = resizedImageInfos[index].ratio;
  if (ratio == 0) {
    LogError << GetError(APP_ERR_DIVIDE_ZERO) << "the ratio must not equal to 0!";
    return;
  }
  TextObjectInfo info;
  info.x0 = NpClip(std::round(box[POINT1].x / ratio), originWidth);
  info.y0 = NpClip(std::round(box[POINT1].y / ratio), originHeight);
  info.x1 = NpClip(std::round(box[POINT2].x / ratio), originWidth);
  info.y1 = NpClip(std::round(box[POINT2].y / ratio), originHeight);
  info.x2 = NpClip(std::round(box[POINT3].x / ratio), originWidth);
  info.y2 = NpClip(std::round(box[POINT3].y / ratio), originHeight);
  info.x3 = NpClip(std::round(box[POINT4].x / ratio), originWidth);
  info.y3 = NpClip(std::round(box[POINT4].y / ratio), originHeight);
  info.confidence = score;

  // check whether current info is valid
  float side1 = std::sqrt(pow((info.x0 - info.x1), INDEX2) + pow((info.y0 - info.y1), INDEX2));
  float side2 = std::sqrt(pow((info.x0 - info.x3), INDEX2) + pow((info.y0 - info.y3), INDEX2));
  float validMinSide = std::max(minSize_ / ratio, minSize_ / ratio);
  if (std::min(side1, side2) < validMinSide) {
    return;
  }
  textObjectInfo->emplace_back(std::move(info));
}

void DBPostprocess::FilterByMinSize(const std::vector<cv::Point> &contour,
                                std::vector<cv::Point2f> *box,
                                float *minSide) {
  cv::Point2f cv_vertices[POINT_NUM];
  cv::RotatedRect cvBox = cv::minAreaRect(contour);
  float width = cvBox.size.width;
  float height = cvBox.size.height;
  *minSide = std::min(width, height);
  cvBox.points(cv_vertices);
  // use vector to manage 4 vertices
  std::vector<cv::Point2f> vertices(cv_vertices, cv_vertices + POINT_NUM);
  // sort the vertices by x-coordinates
  std::sort(vertices.begin(), vertices.end(), SortByX);
  std::sort(vertices.begin(), vertices.begin() + POINT3, SortByY);
  std::sort(vertices.begin() + POINT3, vertices.end(), SortByY);
  // save the box
  box->push_back(vertices[POINT1]);
  box->push_back(vertices[POINT3]);
  box->push_back(vertices[POINT4]);
  box->push_back(vertices[POINT2]);
}

void DBPostprocess::FilterByBoxScore(const cv::Mat &prediction, const std::vector<cv::Point2f> &box, float *score) {
  std::vector<cv::Point2f> tmpbox = box;
  std::sort(tmpbox.begin(), tmpbox.end(), SortByX);

  // construct the mask according to box coordinates.
  int minX = NpClip(static_cast<int>(std::floor(tmpbox.begin()->x)), resizedW_);
  int maxX = NpClip(std::ceil(tmpbox.back().x), resizedW_);
  std::sort(tmpbox.begin(), tmpbox.end(), SortByY);
  int minY = NpClip(static_cast<int>(std::floor(tmpbox.begin()->y)), resizedH_);
  int maxY = NpClip(static_cast<int>(std::ceil(tmpbox.back().y)), resizedH_);
  cv::Mat mask = cv::Mat::zeros(maxY - minY + 1, maxX - minX + 1, CV_8UC1);
  cv::Mat predCrop;
  cv::Point abs_point[POINT_NUM];
  for (int i = 0; i < POINT_NUM; ++i) {
    abs_point[i].x = static_cast<int>(box[i].x - minX);
    abs_point[i].y = static_cast<int>(box[i].y - minY);
  }
  const cv::Point *ppt[1] = {abs_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  // use cv method to calculate the box score
  prediction(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(predCrop);
  *score = cv::mean(predCrop, mask)[0];
}

void DBPostprocess::FilterByClippedMinSize(std::vector<cv::Point2f> *box, float *minSide) {
  // calculate the clip distance
  if (box->size() != POINT_NUM) {
    LogError << "Box size must be four";
    return;
  }
  float side01 = PointsL2Distance((*box)[POINT1], (*box)[POINT2]);
  float side12 = PointsL2Distance((*box)[POINT2], (*box)[POINT3]);
  float side23 = PointsL2Distance((*box)[POINT3], (*box)[POINT4]);
  float side30 = PointsL2Distance((*box)[POINT4], (*box)[POINT1]);
  float diag = PointsL2Distance((*box)[POINT2], (*box)[POINT4]);

  float perimeter = side01 + side12 + side23 + side30;
  float k1 = (side01 + diag + side30) / INDEX2;
  float k2 = (side12 + side23 + diag) / INDEX2;
  float area1 = std::sqrt(k1 * (k1 - side01) * (k1 - diag) * (k1 - side30));
  float area2 = std::sqrt(k2 * (k2 - side12) * (k2 - side23) * (k2 - diag));

  float area = area1 + area2;
  float distance = area * unclipRatio_ / perimeter;

  ClipperLib::ClipperOffset rect;
  ClipperLib::Path path;
  for (auto point : *box) {
    path.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
  }
  rect.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
  ClipperLib::Paths result;
  rect.Execute(result, distance);

  std::vector<cv::Point> contour;
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = 0; j < result[result.size() - 1].size(); ++j) {
      contour.emplace_back(result[i][j].X, result[i][j].Y);
    }
  }
  // check for exception
  box->clear();
  FilterByMinSize(contour, box, minSide);
}

int DBPostprocess::NpClip(const int &coordinate, const int &sideLen) {
  if (coordinate < 0) {
    return 0;
  }
  if (coordinate > sideLen - 1) {
    return sideLen - 1;
  }
  return coordinate;
}

bool DBPostprocess::SortByX(cv::Point2f p1, cv::Point2f p2) {
  return p1.x < p2.x;
}

bool DBPostprocess::SortByY(cv::Point2f p1, cv::Point2f p2) {
  return p1.y < p2.y;
}

float DBPostprocess::PointsL2Distance(cv::Point2f p1, cv::Point2f p2) {
  return std::sqrt(pow((p1.x - p2.x), INDEX2) + pow((p1.y - p2.y), INDEX2));
}
