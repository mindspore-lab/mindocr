#include <securec.h>
#include <algorithm>
#include <string>
#include <vector>
#include "recognition/rec_infer_node.h"
#include "utils/utils.h"
#include "recognition/rec_pre_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_RecInferNode;
const int INT_WIDTH_RATIO = 32;
const float FLOAT_WIDTH_RATIO = 32.0f;
const int BATCH_SIZE_BIAS = 2;

RecPreNode::RecPreNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

RecPreNode::~RecPreNode() = default;

Status RecPreNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;

  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "crnn_pre_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }

  if (staticMethod_) {
    std::string tempPath("./temp/crnn");
    std::vector<std::string> files;
    Utils::GetAllFiles(tempPath, &files);
    for (auto &file : files) {
      std::vector<std::string> nameInfo;
      Utils::StrSplit(file, ".", &nameInfo);
      batchSizeList_.push_back(uint64_t(std::stoi(nameInfo[nameInfo.size() - BATCH_SIZE_BIAS])));
      if (gearInfo_.empty()) {
        Utils::LoadFromFilePair(file, &gearInfo_);
      }
    }

    std::sort(gearInfo_.begin(), gearInfo_.end(), Utils::PairCompare);
    std::sort(batchSizeList_.begin(), batchSizeList_.end(), Utils::UintCompare);
    recMaxWidth_ = gearInfo_[gearInfo_.size() - 1].second;
    recMinWidth_ = gearInfo_[0].second;
    stdHeight_ = gearInfo_[0].first;
  }
  LogInfo << recMinWidth_ << " " << recMaxWidth_ << " " << stdHeight_;
  LogInfo << "crnn_pre_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status RecPreNode::DeInit() {
  LogInfo << "crnn_pre_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status RecPreNode::ParseConfig(CommandParser *options) {
  stdHeight_ = options->GetIntOption("--rec_height");
  LogDebug << "recHeight: " << stdHeight_;

  staticMethod_ = options->GetBoolOption("--static_rec_model_mode");
  LogDebug << "staticRecModelMode: " << staticMethod_;
  taskType_ = Utils::GetTaskType(options);
  if (!staticMethod_) {
    recMinWidth_ = options->GetIntOption("--rec_min_width");
    LogDebug << "recMinWidth: " << recMinWidth_;
    if (recMinWidth_ < 1) {
      LogError << "recMinWidth: " << recMinWidth_ << " is less than 1, not valid";
      return Status::COMM_INVALID_PARAM;
    }
    recMaxWidth_ = std::ceil(static_cast<float>(recMinWidth_) / FLOAT_WIDTH_RATIO) * INT_WIDTH_RATIO;

    recMaxWidth_ = options->GetIntOption("--rec_max_width");
    if (recMaxWidth_ < 1) {
      LogError << "recMaxWidth: " << recMaxWidth_ << " is less than 1, not valid";
      return Status::COMM_INVALID_PARAM;
    }
    recMaxWidth_ = std::floor(static_cast<float>(recMaxWidth_) / FLOAT_WIDTH_RATIO) * INT_WIDTH_RATIO;
    LogDebug << "recMaxWidth: " << recMaxWidth_;
  }
  return Status::OK;
}

void RecPreNode::GetGearInfo(int maxResizedW, std::pair<uint64_t, uint64_t> *gear) {
  if (maxResizedW <= recMaxWidth_) {
    auto info = std::upper_bound(gearInfo_.begin(), gearInfo_.end(),
                                 std::pair<uint64_t, uint64_t>(stdHeight_, maxResizedW), Utils::GearCompare);
    *gear = gearInfo_[info - gearInfo_.begin()];
  }
}

int RecPreNode::GetCrnnMaxWidth(const std::vector<cv::Mat> &frames, float maxWHRatio) {
  int maxResizedW = 0;
  for (auto &frame : frames) {
    int resizedW;
    int imgH = frame.rows;
    int imgW = frame.cols;
    float ratio = imgW / static_cast<float>(imgH);
    int maxWidth = static_cast<int>(maxWHRatio * stdHeight_);
    if (std::ceil(stdHeight_ * ratio) > maxWidth) {
      resizedW = maxWidth;
    } else {
      resizedW = static_cast<int>(std::ceil(stdHeight_ * ratio));
    }
    maxResizedW = std::max(resizedW, maxResizedW);
    maxResizedW = std::max(std::min(maxResizedW, recMaxWidth_), recMinWidth_);
  }
  std::pair<uint64_t, uint64_t> gear;
  if (staticMethod_) {
    GetGearInfo(maxResizedW, &gear);
  } else {
    gear.second = std::ceil(maxResizedW / FLOAT_WIDTH_RATIO) * INT_WIDTH_RATIO;
  }

  return gear.second;
}

uint8_t *RecPreNode::PreprocessCrnn(const std::vector<cv::Mat> &frames, uint32_t batchSize, int maxResizedW,
                                        float maxWHRatio, std::vector<ResizedImageInfo> *resizedImageInfos) {
  cv::Mat resizedImg;
  cv::Mat inImg;
  cv::Mat outImg;
  int resizedW;
  int imgH;
  int imgW;
  uint32_t bufferLen = Utils::RgbImageSizeF32(maxResizedW, stdHeight_);
  uint8_t *srcData = new uint8_t[bufferLen * batchSize];

  int pos = 0;
  for (uint32_t i = 0; i < frames.size(); i++) {
    inImg = frames[i];
    imgH = inImg.rows;
    imgW = inImg.cols;
    float ratio = imgW / static_cast<float >(imgH);
    int maxWidth = static_cast<int>(maxWHRatio * stdHeight_);
    if (std::ceil(stdHeight_ * ratio) > maxWidth) {
      resizedW = maxWidth;
    } else {
      resizedW = static_cast<int>(std::ceil(stdHeight_ * ratio));
    }
    resizedW = std::min(resizedW, recMaxWidth_);
    cv::resize(inImg, resizedImg, cv::Size(resizedW, stdHeight_));
    int paddingLen = maxResizedW - resizedW;
    if (paddingLen > 0) {
      cv::copyMakeBorder(resizedImg, resizedImg, 0, 0, 0, paddingLen, cv::BORDER_CONSTANT, 0);
    }

    LogDebug << "input image [" << i << "] size / preprocessed image size: " << inImg.size() << "/" <<
             resizedImg.size();

    ResizedImageInfo resizedInfo{};
    resizedInfo.widthResize = resizedW;
    resizedInfo.heightResize = stdHeight_;
    resizedInfo.widthOriginal = inImg.cols;
    resizedInfo.heightOriginal = inImg.rows;
    resizedImageInfos->emplace_back(resizedInfo);

    outImg = resizedImg;
    outImg.convertTo(outImg, CV_32FC3, 1.0 / RGB_MAX_IMAGE_VALUE);
    outImg = (outImg - MEAN) / MEAN;

    // GRAY channel means
    std::vector<cv::Mat> channels;
    cv::split(outImg, channels);

    // Transform NHWC to NCHW
    uint32_t size = Utils::RgbImageSizeF32(maxResizedW, stdHeight_);
    uint8_t *buffer = Utils::ImageNchw(channels, size);

    // 把padding后的图片都组装起来
    if (memcpy_s(srcData + pos, bufferLen, buffer, bufferLen) == 0) {
      pos += bufferLen;
      delete[] buffer;
    } else {
      LogError << "memcpy_s failed";
      delete[] buffer;
      continue;
    }
  }

  return srcData;
}

std::vector<uint32_t> RecPreNode::GetCrnnBatchSize(uint32_t frameSize) {
  int lastIndex = batchSizeList_.size() - 1;
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

Status RecPreNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  if (taskType_ == TaskType::REC) {
    data->imgMatVec.push_back(data->frame);
  }
  uint32_t totalSize = data->imgMatVec.size();
  if (totalSize == 0) {
    data->eof = true;
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Profile::recPreProcessTime_ += costTime;
    Profile::e2eProcessTime_ += costTime;
    SendToNextModule(MT_RecInferNode, data, data->channelId);
    return Status::OK;
  }

  std::vector<uint32_t> splitIndex = {totalSize};
  if (staticMethod_) {
    splitIndex = GetCrnnBatchSize(totalSize);
  }

  int startIndex = 0;
  int shareId = 0;
  for (unsigned int i : splitIndex) {
    std::shared_ptr<CommonData> dataNew = std::make_shared<CommonData>();

    std::vector<ResizedImageInfo> resizedImageInfosCrnn;
    std::vector<cv::Mat> input(data->imgMatVec.begin() + startIndex,
                               data->imgMatVec.begin() + std::min(startIndex + i, totalSize));
    if (taskType_ == TaskType::DET_CLS_REC || taskType_ == TaskType::DET_REC) {
      std::vector<std::string> splitRes(data->inferRes.begin() + startIndex,
                                        data->inferRes.begin() + std::min(startIndex + i, totalSize));
      dataNew->inferRes = splitRes;
    }
    int maxResizedW = GetCrnnMaxWidth(input, data->maxWHRatio);
    if (taskType_ == TaskType::REC) {
      data->subImgTotal += 1;
    }
    uint8_t *crnnInput = PreprocessCrnn(input, i, maxResizedW, data->maxWHRatio, &resizedImageInfosCrnn);
    shareId++;

    dataNew->eof = false;
    dataNew->outputMindXTensorVec = data->outputMindXTensorVec;
    dataNew->outputLiteTensorVec = data->outputLiteTensorVec;
    dataNew->backendType = data->backendType;
    dataNew->imgName = data->imgName;
    dataNew->imgTotal = data->imgTotal;
    dataNew->maxResizedW = maxResizedW;

    dataNew->resizedImageInfos = resizedImageInfosCrnn;
    dataNew->batchSize = i;
    dataNew->imgBuffer = crnnInput;
    dataNew->imageName = data->imageName;
    dataNew->frameSize = std::min(startIndex + i, totalSize) - startIndex;
    dataNew->subImgTotal = data->subImgTotal;
    dataNew->imgId = data->imgId;

    startIndex += i;
    SendToNextModule(MT_RecInferNode, dataNew, data->channelId);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::recPreProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;

  return Status::OK;
}
