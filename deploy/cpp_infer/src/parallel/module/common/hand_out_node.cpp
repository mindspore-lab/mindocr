#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <memory>
#include <regex>
#include "detection/det_pre_node.h"
#include "classification/cls_pre_node.h"
#include "recognition/rec_pre_node.h"
#include "common/hand_out_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_DetPreNode;
using AscendBaseModule::MT_RecPreNode;
using AscendBaseModule::MT_ClsPreNode;

HandoutNode::HandoutNode() {
  withoutInputQueue_ = true;
  isStop_ = false;
}

HandoutNode::~HandoutNode() = default;

Status HandoutNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;
  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "hand_out_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }
  deviceType_ = options->GetStringOption("--device_type");
  if (deviceType_ != "310P" && deviceType_ != "310") {
    LogError << "Device type only support 310 or 310P, please check the value of device type.";
    return Status::COMM_INVALID_PARAM;
  }
  if (deviceType_ == "310P") {
    imageProcessor_.reset(new MxBase::ImageProcessor(deviceId_));
  }
  TaskType taskType_ = Utils::GetTaskType(options);
  if (taskType_ == TaskType::DET || taskType_ == TaskType::DET_REC || taskType_ == TaskType::DET_CLS_REC) {
    nextModule_ = MT_DetPreNode;
  } else if (taskType_ == TaskType::REC) {
    nextModule_ = MT_RecPreNode;
  } else if (taskType_ == TaskType::CLS) {
    nextModule_ = MT_ClsPreNode;
  }
  LogInfo << "hand_out_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status HandoutNode::DeInit() {
  LogInfo << "hand_out_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status HandoutNode::ParseConfig(CommandParser *options) {
  resultPath_ = options->GetStringOption("--res_save_dir");
  if (resultPath_.empty()) {
    return Status::COMM_INVALID_PARAM;
  }
  backendType_ = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  return Status::OK;
}

cv::Mat HandoutNode::DecodeImgDvpp(std::string imgPath) {
  MxBase::Image decodedImage;
  imageProcessor_->Decode(imgPath, decodedImage, MxBase::ImageFormat::BGR_888);
  decodedImage.ToHost();

  MxBase::Size imgOriSize = decodedImage.GetOriginalSize();
  MxBase::Size imgSize = decodedImage.GetSize();
  cv::Mat imgBGR;
  imgBGR.create(imgSize.height, imgSize.width, CV_8UC3);
  imgBGR.data = reinterpret_cast<uchar *>(decodedImage.GetData().get());
  cv::Rect area(0, 0, imgOriSize.width, imgOriSize.height);
  imgBGR = imgBGR(area).clone();
  return imgBGR;
}

Status HandoutNode::Process(std::shared_ptr<void> commonData) {
  std::string configPath = "config";
  if (access(configPath.c_str(), 0) == -1) {
    int retCode = system(("mkdir -p " + configPath).c_str());
    if (retCode == -1) {
      LogError << "Can not create dir [" << configPath << "], please check the value of config path";
      return Status::COMM_INVALID_PARAM;
    }
    LogInfo << configPath << " create!";
  }
  std::string imgConfig = "./config/" + pipelineName_;
  LogInfo << pipelineName_;
  std::ifstream imgFileCount;
  imgFileCount.open(imgConfig);
  std::string imgPathCount;
  int imgTotal = 0;
  while (getline(imgFileCount, imgPathCount)) {
    imgTotal++;
  }
  imgFileCount.close();

  std::ifstream imgFile;
  imgFile.open(imgConfig);
  std::string imgPath;
  std::regex reg("^([A-Za-z]+)_([0-9]+).*$");
  std::cmatch m;
  std::string basename;
  while (getline(imgFile, imgPath) && !Profile::signalReceived_) {
    LogInfo << pipelineName_ << " read file:" << imgPath;
    basename = Utils::BaseName(imgPath);
    std::regex_match(basename.c_str(), m, reg);
    if (m.empty()) {
      LogError << "Please check the image name format of " << basename <<
               ". the image name should be xxx_xxx.xxx";
      continue;
    }
    imgId_++;
    std::shared_ptr<CommonData> data = std::make_shared<CommonData>();
    data->imgPath = imgPath;
    data->imgId = imgId_;
    data->imgTotal = imgTotal;
    data->imgName = basename;
    data->imageName = Utils::GetImageName(imgPath);
    data->backendType = backendType_;
    cv::Mat inImg;
    if (deviceType_ == "310P") {
      inImg = DecodeImgDvpp(imgPath);
    } else {
      inImg = cv::imread(imgPath);
    }
    data->frame = inImg;
    data->srcWidth = inImg.cols;
    data->srcHeight = inImg.rows;
    SendToNextModule(nextModule_, data, data->channelId);
  }

  imgFile.close();
  return Status::OK;
}
