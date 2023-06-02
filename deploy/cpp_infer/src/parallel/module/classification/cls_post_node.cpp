#include <vector>
#include "MxBase/MxBase.h"
#include "utils/utils.h"
#include "common/collect_node.h"
#include "recognition/rec_pre_node.h"
#include "classification/cls_post_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_CollectNode;
using AscendBaseModule::MT_RecPreNode;

ClsPostNode::ClsPostNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

ClsPostNode::~ClsPostNode() = default;

Status ClsPostNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;

  AssignInitArgs(initArgs);
  taskType_ = Utils::GetTaskType(options);
  if (taskType_ == TaskType::CLS) {
    nextModule_ = MT_CollectNode;
  } else if (taskType_ == TaskType::DET_CLS_REC) {
    nextModule_ = MT_RecPreNode;
  } else {
    LogError << "Unsupported task type";
    return Status::UNSUPPORTED_TASK_TYPE;
  }
  LogInfo << "cls_post_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status ClsPostNode::DeInit() {
  LogInfo << "cls_post_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status ClsPostNode::PostProcessMindXCls(uint32_t framesSize, const std::vector<MxBase::Tensor> &inferOutput,
                                        const std::vector<cv::Mat> imgMatVec, std::vector<std::string> *inferRes) {
  std::vector<uint32_t> shape = inferOutput[0].GetShape();
  auto *tensorData = reinterpret_cast<float *>(inferOutput[0].GetData());
  std::vector<int64_t> tmpShape(shape.begin(), shape.end());
  GenerateInferResAndRotate(framesSize, imgMatVec, tmpShape, tensorData, inferRes);
  return Status::OK;
}

Status ClsPostNode::PostProcessLiteCls(uint32_t framesSize, const std::vector<mindspore::MSTensor> &inferOutput,
                                       const std::vector<cv::Mat> &imgMatVec, std::vector<std::string> *inferRes) {
  std::vector<int64_t> shape = inferOutput[0].Shape();
  auto *tensorData = reinterpret_cast<const float *>(inferOutput[0].Data().get());
  GenerateInferResAndRotate(framesSize, imgMatVec, shape, tensorData, inferRes);
  return Status::OK;
}

void
ClsPostNode::GenerateInferResAndRotate(uint32_t framesSize, const std::vector<cv::Mat> &imgMatVec,
                                       const std::vector<int64_t> &shape,
                                       const float *tensorData, std::vector<std::string> *inferRes) {
  uint32_t dirVecSize = shape[1];
  for (uint32_t i = 0; i < framesSize; i++) {
    uint32_t zeroDegreeIndex = i * dirVecSize + 0;
    uint32_t oneHundredEightyDegreeIndex = i * dirVecSize + 1;
    if (taskType_ == TaskType::CLS) {
      std::string tmpRes;
      tmpRes.append("[\"");
      if (tensorData[zeroDegreeIndex] > tensorData[oneHundredEightyDegreeIndex]) {
        tmpRes.append("0\", ").append(std::to_string(tensorData[zeroDegreeIndex]));
      } else {
        tmpRes.append("180\", ").append(std::to_string(tensorData[oneHundredEightyDegreeIndex]));
      }
      tmpRes.append("]");
      inferRes->push_back(tmpRes);
    }
    if (tensorData[oneHundredEightyDegreeIndex] > ROTATE_THRESH) {
      cv::rotate(imgMatVec[i], imgMatVec[i], cv::ROTATE_180);
    }
  }
}

Status ClsPostNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);

  Status ret;
  if (!data->eof) {
    if (data->backendType == BackendType::ACL) {
      ret = PostProcessMindXCls(data->frameSize, data->outputMindXTensorVec, data->imgMatVec, &data->inferRes);
    } else if (data->backendType == BackendType::LITE) {
      ret = PostProcessLiteCls(data->frameSize, data->outputLiteTensorVec, data->imgMatVec, &data->inferRes);
    } else {
      ret = Status::UNSUPPORTED_INFER_ENGINE;
    }
    if (ret != Status::OK) {
      return ret;
    }
  }
  SendToNextModule(nextModule_, data, data->channelId);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::clsPostProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  return Status::OK;
}
