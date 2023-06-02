#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include "classification/cls_infer_node.h"
#include "utils/utils.h"
#include "data_type/constant.h"
#include "classification/cls_pre_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_ClsInferNode;

ClsPreNode::ClsPreNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

ClsPreNode::~ClsPreNode() = default;

Status ClsPreNode::Init(CommandParser *options, const ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;

  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "cls_pre_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }
  clsHeight_ = DEFAULT_CLS_HEIGHT;
  clsWidth_ = DEFAULT_CLS_WIDTH;

  LogInfo << "cls_pre_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status ClsPreNode::DeInit() {
  LogInfo << "cls_pre_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status ClsPreNode::ParseConfig(CommandParser *options) {
  std::string clsModelPath = options->GetStringOption("--cls_model_path");
  std::string baseName = Utils::BaseName(clsModelPath) + ".bin";
  std::string modelConfigPath("./temp/cls/");
  Utils::LoadFromFileVec(modelConfigPath + baseName, &batchSizeList_);
  taskType_ = Utils::GetTaskType(options);
  if (taskType_ == TaskType::UNSUPPORTED) {
    LogError << "Unsupported task type";
    return Status::UNSUPPORTED_TASK_TYPE;
  }
  return Status::OK;
}

uint8_t *ClsPreNode::PreprocessCls(const std::vector<cv::Mat> &frames, uint32_t batchSize) {
  cv::Mat resizedImg;
  cv::Mat inImg;

  uint32_t bufferLen = Utils::RgbImageSizeF32(clsWidth_, clsHeight_);
  auto *srcData = new uint8_t[bufferLen * batchSize];

  int pos = 0;
  for (uint32_t i = 0; i < frames.size(); i++) {
    inImg = frames[i];
    float ratio = static_cast<float>(inImg.cols) / static_cast<float>(inImg.rows);
    int resize_w;
    if (std::ceil(clsHeight_ * ratio) > clsWidth_)
      resize_w = clsWidth_;
    else
      resize_w = static_cast<int>(std::ceil(clsHeight_ * ratio));
    cv::resize(inImg, resizedImg, cv::Size(resize_w, clsHeight_), 0.f, 0.f, cv::INTER_LINEAR);
    if (resize_w < clsWidth_) {
      cv::copyMakeBorder(resizedImg, resizedImg, 0, 0, 0, clsWidth_ - resize_w, cv::BORDER_CONSTANT,
                         cv::Scalar(0, 0, 0));
    }

    resizedImg.convertTo(resizedImg, CV_32FC3, 1.0 / RGB_MAX_IMAGE_VALUE);
    resizedImg = (resizedImg - MEAN) / MEAN;

    std::vector<cv::Mat> channels;
    cv::split(resizedImg, channels);

    // Transform NHWC to NCHW
    uint32_t size = Utils::RgbImageSizeF32(clsWidth_, clsHeight_);
    uint8_t *buffer = Utils::ImageNchw(channels, size);

    // 把padding后的图片都组装起来
    if (memcpy_s(srcData + pos, bufferLen, buffer, bufferLen) == 0) {
      pos += bufferLen;
    } else {
      LogError << "memcpy_s failed";
    }
    delete[] buffer;
  }

  return srcData;
}

std::vector<uint32_t> ClsPreNode::GetClsBatchSize(uint32_t frameSize) {
  auto lastIndex = batchSizeList_.size() - 1;
  std::vector<uint32_t> splitList(frameSize / batchSizeList_[lastIndex], batchSizeList_[lastIndex]);
  frameSize = frameSize - batchSizeList_[lastIndex] * (frameSize / batchSizeList_[lastIndex]);
  if (!frameSize) {
    return splitList;
  }
  for (auto bs : batchSizeList_) {
    if (frameSize <= bs) {
      splitList.push_back(bs);
      break;
    }
  }
  return splitList;
}

Status ClsPreNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  if (taskType_ == TaskType::CLS) {
    data->imgMatVec.push_back(data->frame);
  }
  uint32_t totalSize = data->imgMatVec.size();
  if (totalSize == 0) {
    data->eof = true;
    SendToNextModule(MT_ClsInferNode, data, data->channelId);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Profile::clsPreProcessTime_ += costTime;
    Profile::e2eProcessTime_ += costTime;
    return Status::OK;
  }

  std::vector<uint32_t> splitIndex = GetClsBatchSize(totalSize);
  int startIndex = 0;
  for (unsigned int &bs : splitIndex) {
    std::shared_ptr<CommonData> dataNew = std::make_shared<CommonData>();

    std::vector<ResizedImageInfo> resizedImageInfosCls;
    std::vector<cv::Mat> input(data->imgMatVec.begin() + startIndex,
                               data->imgMatVec.begin() + std::min(startIndex + bs, totalSize));

    if (taskType_ == TaskType::DET_CLS_REC) {
      std::vector<std::string> splitRes(data->inferRes.begin() + startIndex,
                                        data->inferRes.begin() + std::min(startIndex + bs, totalSize));
      dataNew->inferRes = splitRes;
    }

    if (taskType_ == TaskType::CLS) {
      data->subImgTotal += 1;
    }

    uint8_t *ClsInput = PreprocessCls(input, bs);

    dataNew->eof = false;
    dataNew->outputMindXTensorVec = data->outputMindXTensorVec;
    dataNew->outputLiteTensorVec = data->outputLiteTensorVec;
    dataNew->imgName = data->imgName;
    dataNew->imgTotal = data->imgTotal;
    dataNew->imgMatVec = input;

    dataNew->resizedImageInfos = resizedImageInfosCls;
    dataNew->batchSize = bs;
    dataNew->imgBuffer = ClsInput;
    dataNew->frameSize = std::min(startIndex + bs, totalSize) - startIndex;
    dataNew->maxWHRatio = data->maxWHRatio;
    dataNew->imageName = data->imageName;
    dataNew->subImgTotal = data->subImgTotal;
    dataNew->imgId = data->imgId;
    dataNew->backendType = data->backendType;

    startIndex += bs;
    SendToNextModule(MT_ClsInferNode, dataNew, data->channelId);
  }
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::clsPreProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  return Status::OK;
}
