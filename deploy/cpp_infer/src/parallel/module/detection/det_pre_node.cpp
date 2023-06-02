#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <memory>
#include <utility>
#include "MxBase/MxBase.h"
#include "data_type/constant.h"
#include "detection/det_infer_node.h"
#include "detection/det_pre_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_DetInferNode;

DetPreNode::DetPreNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

DetPreNode::~DetPreNode() = default;

Status DetPreNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;
  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "dbnet_pre_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }

  if (deviceType_ == "310P") {
    imageProcessor_ = std::make_unique<MxBase::ImageProcessor>(deviceId_);
  }
  LogInfo << "dbnet_pre_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status DetPreNode::DeInit() {
  LogInfo << "dbnet_pre_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status DetPreNode::ParseConfig(CommandParser *options) {
  deviceType_ = options->GetStringOption("--device_type");
  if (deviceType_ != "310P" && deviceType_ != "310") {
    LogError << "Device type only support 310 or 310P, please check the value of device type.";
    return Status::COMM_INVALID_PARAM;
  }

  std::vector<uint32_t> deviceIdVec;
  auto ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK || deviceIdVec.empty()) {
    LogError << "Get device id failed, please check the value of deviceId";
    return Status::COMM_INVALID_PARAM;
  }
  deviceId_ = (int32_t) deviceIdVec[instanceId_ % deviceIdVec.size()];
  if (deviceId_ < 0) {
    LogError << "Device id: " << deviceId_ << " is less than 0, not valid";
    return Status::COMM_INVALID_PARAM;
  }

  std::string detModelPath = options->GetStringOption("--det_model_path");
  std::string baseName = Utils::BaseName(detModelPath) + ".1.bin";
  std::string modelConfigPath("./temp/dbnet/");
  Utils::LoadFromFilePair(modelConfigPath + baseName, &gearInfo_);
  std::sort(gearInfo_.begin(), gearInfo_.end(), Utils::PairCompare);
  uint64_t hwSum = 0;
  for (auto &pair : gearInfo_) {
    uint64_t h = pair.first;
    uint64_t w = pair.second;
    maxH_ = maxH_ > h ? maxH_ : h;
    maxW_ = maxW_ > w ? maxW_ : w;
    if (h * w > hwSum) {
      hwSum = h * w;
      maxDotGear_.first = h;
      maxDotGear_.second = w;
    }
  }

  return Status::OK;
}

void DetPreNode::GetMatchedGear(const cv::Mat &inImg, std::pair<uint64_t, uint64_t> *gear) {
  uint64_t imgH = inImg.rows;
  uint64_t imgW = inImg.cols;
  if (imgH > maxH_ || imgW > maxW_) {
    *gear = maxDotGear_;
  } else {
    auto info = std::upper_bound(gearInfo_.begin(), gearInfo_.end(), std::pair<uint64_t, uint64_t>(imgH, imgW),
                                 Utils::GearCompare);
    *gear = gearInfo_[info - gearInfo_.begin()];
  }
}

void DetPreNode::Resize(const cv::Mat &inImg, cv::Mat *outImg, const std::pair<uint64_t, uint64_t> &gear,
                        float *inputRatio) {
  int imgH = inImg.rows;
  int imgW = inImg.cols;
  int gearH = gear.first;
  int gearW = gear.second;
  float ratio = 1.f;
  if (imgH > gearH || imgW > gearW) {
    if (imgH > imgW) {
      ratio = static_cast<float>(gearH) / static_cast<float>(imgH);
      int resizeByH = static_cast<int>(ratio * static_cast<float>(imgW));
      if (resizeByH > gearW) {
        ratio = static_cast<float>(gearW) / static_cast<float>(imgW);
      }
    } else {
      ratio = static_cast<float>(gearW) / static_cast<float>(imgW);
      int resizeByW = static_cast<int>(ratio * static_cast<float>(imgH));
      if (resizeByW > gearH) {
        ratio = static_cast<float>(gearH) / static_cast<float>(imgH);
      }
    }
  }
  int resizeH = static_cast<int>(static_cast<float>(imgH) * ratio);
  int resizeW = static_cast<int>(static_cast<float>(imgW) * ratio);
  cv::resize(inImg, *outImg, cv::Size(resizeW, resizeH));
  *inputRatio = static_cast<float>(resizeH) / static_cast<float>(imgH);
}

void DetPreNode::Padding(cv::Mat *inImg, const std::pair<uint64_t, uint64_t> &gear) {
  int imgH = inImg->rows;
  int imgW = inImg->cols;
  int gearH = gear.first;
  int gearW = gear.second;
  int paddingH = gearH - imgH;
  int paddingW = gearW - imgW;
  if (paddingH || paddingW) {
    cv::copyMakeBorder(*inImg, *inImg, 0, paddingH, 0, paddingW, cv::BORDER_CONSTANT, 0);
  }
}

void DetPreNode::NormalizeByChannel(std::vector<cv::Mat> *bgrChannels) {
  for (uint32_t i = 0; i < bgrChannels->size(); i++) {
    (*bgrChannels)[i].convertTo((*bgrChannels)[i], CV_32FC1, 1.0 * scale_[i], (0.0 - mean_[i]) * scale_[i]);
  }
}

Status DetPreNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  std::string imgPath = data->imgPath;

  std::chrono::high_resolution_clock::time_point dbPreEndTime;
  cv::Mat inImg = data->frame;
  cv::Mat resizedImg;
  cv::Mat outImg;

  inImg.convertTo(inImg, CV_32FC3, 1.0 / RGB_MAX_IMAGE_FLOAT_VALUE);
  std::pair<uint64_t, uint64_t> gear;
  GetMatchedGear(inImg, &gear);

  float ratio = 0;
  Resize(inImg, &resizedImg, gear, &ratio);

  Padding(&resizedImg, gear);

  // Normalize: y = (x - mean) / std
  std::vector<cv::Mat> bgrChannels(CHANNEL_SIZE);
  cv::split(resizedImg, bgrChannels);
  NormalizeByChannel(&bgrChannels);

  // Transform NHWC to NCHW
  uint32_t size = Utils::RgbImageSizeF32(resizedImg.cols, resizedImg.rows);
  uint8_t *buffer = Utils::ImageNchw(bgrChannels, size);

  data->eof = false;
  data->channelId = 0;
  data->imgBuffer = buffer;
  data->resizeWidth = resizedImg.cols;
  data->resizeHeight = resizedImg.rows;
  data->ratio = ratio;
  SendToNextModule(MT_DetInferNode, data, data->channelId);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::detPreProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;

  return Status::OK;
}
