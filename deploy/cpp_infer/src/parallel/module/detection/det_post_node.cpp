#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#include "common/collect_node.h"
#include "recognition/rec_pre_node.h"
#include "classification/cls_pre_node.h"
#include "utils/utils.h"
#include "postprocess/db_postprocess.h"
#include "detection/det_post_node.h"

const float IMAGE_WIDTH_HEIGHT_RATIO = 1.5f;

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_CollectNode;
using AscendBaseModule::MT_RecPreNode;
using AscendBaseModule::MT_ClsPreNode;

DetPostNode::DetPostNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

DetPostNode::~DetPostNode() = default;

Status DetPostNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;
  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "dbnet_post_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }
  TaskType taskType = Utils::GetTaskType(options);
  if (taskType == TaskType::DET) {
    nextModule_ = MT_CollectNode;
  } else if (taskType == TaskType::DET_REC) {
    nextModule_ = MT_RecPreNode;
  } else if (taskType == TaskType::DET_CLS_REC) {
    nextModule_ = MT_ClsPreNode;
  }
  LogInfo << "dbnet_post_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status DetPostNode::DeInit() {
  LogInfo << "dbnet_post_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status DetPostNode::ParseConfig(CommandParser *options) {
  resultPath_ = options->GetStringOption("--res_save_dir");
  if (access(resultPath_.c_str(), 0) == -1) {
    int retCode = system(("mkdir -p " + resultPath_).c_str());
    if (retCode == -1) {
      LogError << "Can not create dir [" << resultPath_ << "], please check the value of resultPath.";
      return Status::COMM_INVALID_PARAM;
    }
    LogInfo << resultPath_ << " create!";
  }
  return Status::OK;
}

Status DetPostNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);

  std::vector<ResizedImageInfo> resizedImageInfos;
  ResizedImageInfo resizedInfo{};

  resizedInfo.widthResize = data->resizeWidth;
  resizedInfo.heightResize = data->resizeHeight;
  resizedInfo.widthOriginal = data->srcWidth;
  resizedInfo.heightOriginal = data->srcHeight;
  resizedInfo.ratio = data->ratio;
  resizedImageInfos.emplace_back(resizedInfo);

  std::vector<std::vector<TextObjectInfo>> textObjInfos;

  DBPostprocess dbnetPost;

  if (data->backendType == BackendType::ACL) {
    dbnetPost.DbnetMindXObjectDetectionOutput(data->outputMindXTensorVec, &textObjInfos, resizedImageInfos);
  } else if (data->backendType == BackendType::LITE) {
    dbnetPost.DbnetLiteObjectDetectionOutput(data->outputLiteTensorVec, &textObjInfos, resizedImageInfos);
  } else {
    LogError << "Unsupported infer engine";
    return Status::UNSUPPORTED_INFER_ENGINE;
  }

  std::vector<cv::Mat> resizeImgs;
  std::vector<std::string> inferRes;
  float maxWHRatio = 0;

  for (const auto &textInfo : textObjInfos) {
    for (auto &j : textInfo) {
      cv::Mat resizeimg;
      std::string str = std::to_string(static_cast<int>(j.x1)) + "," + std::to_string(static_cast<int>(j.y1)) + "," +
          std::to_string(static_cast<int>(j.x2)) + "," + std::to_string(static_cast<int>(j.y2)) + "," +
          std::to_string(static_cast<int>(j.x3)) + "," + std::to_string(static_cast<int>(j.y3)) + "," +
          std::to_string(static_cast<int>(j.x0)) + "," + std::to_string(static_cast<int>(j.y0)) + ",";
      inferRes.push_back(str);

      float cropWidth = CalcCropWidth(j);
      float cropHeight = CalcCropHeight(j);

      // 期望透视变换后二维码四个角点的坐标
      cv::Point2f dstPoints[4];
      cv::Point2f srcPoints[4];
      // 通过Image Watch查看的二维码四个角点坐标
      srcPoints[POINT1] = cv::Point2f(j.x0, j.y0);
      srcPoints[POINT2] = cv::Point2f(j.x1, j.y1);
      srcPoints[POINT3] = cv::Point2f(j.x2, j.y2);
      srcPoints[POINT4] = cv::Point2f(j.x3, j.y3);

      dstPoints[POINT1] = cv::Point2f(0.0, 0.0);
      dstPoints[POINT2] = cv::Point2f(cropWidth, 0.0);
      dstPoints[POINT3] = cv::Point2f(cropWidth, cropHeight);
      dstPoints[POINT4] = cv::Point2f(0.0, cropHeight);

      cv::Mat H = cv::getPerspectiveTransform(srcPoints, dstPoints);

      cv::Mat rotation;

      cv::Mat img_warp = cv::Mat(cropHeight, cropWidth, CV_8UC3);
      cv::warpPerspective(data->frame, img_warp, H, img_warp.size());
      int imgH = img_warp.rows;
      int imgW = img_warp.cols;
      if (imgH * 1.0 / imgW >= IMAGE_WIDTH_HEIGHT_RATIO) {
        cv::rotate(img_warp, img_warp, cv::ROTATE_90_COUNTERCLOCKWISE);
        imgH = img_warp.rows;
        imgW = img_warp.cols;
      }
      maxWHRatio = std::max(maxWHRatio, static_cast<float>(imgW) / static_cast<float>(imgH));
      resizeImgs.emplace_back(img_warp);
    }
  }
  data->maxWHRatio = maxWHRatio;
  data->imgMatVec = resizeImgs;
  data->inferRes = inferRes;
  data->subImgTotal = resizeImgs.size();
  if (data->inferRes.empty()) {
    LogError << data->imageName;
    LogError << "dbnet_post_process inferRes empty";
  }
  SendToNextModule(nextModule_, data, data->channelId);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::detPostProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  return Status::OK;
}

float DetPostNode::CalcCropWidth(const TextObjectInfo &textObject) {
  float x0 = std::abs(textObject.x1 - textObject.x0);
  float y0 = std::abs(textObject.y1 - textObject.y0);
  float line0 = sqrt(std::pow(x0, 2) + std::pow(y0, 2));

  float x1 = std::abs(textObject.x2 - textObject.x3);
  float y1 = std::abs(textObject.y2 - textObject.y3);
  float line1 = std::sqrt(std::pow(x1, 2) + std::pow(y1, 2));

  return line1 > line0 ? line1 : line0;
}

float DetPostNode::CalcCropHeight(const TextObjectInfo &textObject) {
  float x0 = std::abs(textObject.x0 - textObject.x3);
  float y0 = std::abs(textObject.y0 - textObject.y3);
  float line0 = sqrt(std::pow(x0, 2) + std::pow(y0, 2));

  float x1 = std::abs(textObject.x1 - textObject.x2);
  float y1 = std::abs(textObject.y1 - textObject.y2);
  float line1 = std::sqrt(std::pow(x1, 2) + std::pow(y1, 2));

  return line1 > line0 ? line1 : line0;
}
