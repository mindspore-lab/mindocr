#include <memory>
#include <string>
#include <vector>
#include "common//collect_node.h"
#include "MxBase/MxBase.h"
#include "utils/utils.h"
#include "recognition/rec_post_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_CollectNode;

RecPostNode::RecPostNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

RecPostNode::~RecPostNode() = default;

Status RecPostNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;

  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "crnn_post_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }
  recCtcLabelDecode_.ClassNameInit(recDictionary_);

  LogInfo << "crnn_post_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status RecPostNode::DeInit() {
  LogInfo << "crnn_post_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status RecPostNode::ParseConfig(CommandParser *options) {
  recDictionary_ = options->GetStringOption("--character_dict_path");
  Status ret = Utils::CheckPath(recDictionary_, "character label file");
  if (ret != Status::OK) {
    LogError << "Character label file: " << recDictionary_ << " is not exist of can not read.";
    return Status::COMM_INVALID_PARAM;
  }
  LogDebug << "dictPath: " << recDictionary_;

  resultPath_ = options->GetStringOption("--res_save_dir");
  if (resultPath_[resultPath_.size() - 1] != '/') {
    resultPath_ += "/";
  }
  return Status::OK;
}

Status RecPostNode::PostProcessMindXCrnn(uint32_t framesSize, const std::vector<MxBase::Tensor> &inferOutput,
                                         std::vector<std::string> *textsInfos) {
  auto *objectInfo = reinterpret_cast<int64_t *>(inferOutput[0].GetData());
  auto objectNum = static_cast<size_t> ( inferOutput[0].GetShape()[1]);
  recCtcLabelDecode_.CalcMindXOutputIndex(objectInfo, framesSize, objectNum, textsInfos);
  return Status::OK;
}

Status RecPostNode::PostProcessLiteCrnn(uint32_t framesSize, const std::vector<mindspore::MSTensor> &inferOutput,
                                        std::vector<std::string> *textsInfos) {
  auto *objectInfo = reinterpret_cast<const int32_t *>(inferOutput[0].Data().get());
  auto objectNum = static_cast<size_t>(inferOutput[0].Shape()[1]);
  recCtcLabelDecode_.CalcLiteOutputIndex(objectInfo, framesSize, objectNum, textsInfos);
  return Status::OK;
}

Status RecPostNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  std::vector<std::string> recResVec;
  if (!data->eof) {
    Status ret;
    if (data->backendType == BackendType::ACL) {
      ret = PostProcessMindXCrnn(data->frameSize, data->outputMindXTensorVec, &data->inferRes);
    } else if (data->backendType == BackendType::LITE) {
      ret = PostProcessLiteCrnn(data->frameSize, data->outputLiteTensorVec, &data->inferRes);
    } else {
      ret = Status::UNSUPPORTED_INFER_ENGINE;
    }
    if (ret != Status::OK) {
      return ret;
    }
  }
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

  Profile::recPostProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  SendToNextModule(MT_CollectNode, data, data->channelId);
  return Status::OK;
}
