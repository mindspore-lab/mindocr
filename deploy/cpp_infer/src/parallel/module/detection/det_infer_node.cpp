#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "data_type/constant.h"
#include "detection/det_pre_node.h"
#include "detection/det_post_node.h"
#include "detection/det_infer_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_DetPostNode;

DetInferNode::DetInferNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

DetInferNode::~DetInferNode() = default;

Status DetInferNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;
  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options, initArgs);
  backend_ = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  if (ret != Status::OK) {
    LogError << "dbnet_infer_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }

  LogInfo << "dbnet_infer_process [" << instanceId_ << "] Init success.";
  return ret;
}

Status DetInferNode::DeInit() {
  LogInfo << "dbnet_infer_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status DetInferNode::ParseConfig(CommandParser *options, const ModuleInitArgs &initArgs) {
  std::vector<uint32_t> deviceIdVec;
  Status ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK || deviceIdVec.empty()) {
    LogError << "Get device id failed, please check the value of deviceId";
    return Status::COMM_INVALID_PARAM;
  }
  deviceId_ = (int32_t) deviceIdVec[instanceId_ % deviceIdVec.size()];
  std::string detModelPath = options->GetStringOption("--det_model_path");
  ret = Utils::CheckPath(detModelPath, "detModelPath");
  if (ret != Status::OK) {
    LogError << "Get detModelPath failed, please check the value of detModelPath";
    return Status::COMM_INVALID_PARAM;
  }
  LogDebug << "detModelPath: " << detModelPath;
  backend_ = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  if (backend_ == BackendType::ACL) {
    dbNetMindX_ = std::make_unique<MxBase::Model>(detModelPath, deviceId_);
  } else if (backend_ == BackendType::LITE) {
    dbNetLite_ = new mindspore::Model();
    auto buildRet = dbNetLite_->Build(detModelPath, mindspore::kMindIR, initArgs.context);
    if (buildRet != mindspore::kSuccess) {
      LogError << "Build lite model error " << buildRet;
      return Status::LITE_MODEL_BUILD_FAILURE;
    }
  } else {
    return Status::UNSUPPORTED_INFER_ENGINE;
  }
  return Status::OK;
}

Status DetInferNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  Status ret;

  if (backend_ == BackendType::ACL) {
    ret = MindXModelInfer(data);
  } else if (backend_ == BackendType::LITE) {
    ret = LiteModelInfer(data);
  } else {
    ret = Status::UNSUPPORTED_INFER_ENGINE;
  }
  if (ret != Status::OK) {
    return ret;
  }
  SendToNextModule(MT_DetPostNode, data, data->channelId);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::detInferProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  return Status::OK;
}

Status DetInferNode::LiteModelInfer(std::shared_ptr<CommonData> data) {
  std::vector<int64_t> shape;
  uint32_t batchSize = 1;
  shape.push_back(batchSize);
  shape.push_back(CHANNEL_SIZE);
  shape.push_back(data->resizeHeight);
  shape.push_back(data->resizeWidth);
  std::vector<std::vector<int64_t>> dims;
  dims.push_back(shape);
  // (2) 开始推理
  auto inferStartTime = std::chrono::high_resolution_clock::now();
  auto inputs = dbNetLite_->GetInputs();
  dbNetLite_->Resize(inputs, dims);
  inputs[0].SetData(data->imgBuffer);
  std::vector<mindspore::MSTensor> outputs;
  auto predictResult = dbNetLite_->Predict(inputs, &outputs);
  if (predictResult != mindspore::kSuccess) {
    LogError << "Lite predict error";
    return Status::LITE_MODEL_INFER_FAILURE;
  }
  auto inferEndTime = std::chrono::high_resolution_clock::now();
  double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
  Profile::detInferTime_ += inferCostTime;
  LogInfo << " [" << data->imgName << "] dbnet infer time cost: " << inferCostTime << "ms.";

  const size_t outputLen = outputs.size();
  if (outputLen <= 0) {
    LogError << "Failed to get model output data";
    return Status::LITE_MODEL_INFER_FAILURE;
  }
  data->outputLiteTensorVec = outputs;
  data->backendType = BackendType::LITE;
  if (data->imgBuffer != nullptr) {
    delete data->imgBuffer;
    data->imgBuffer = nullptr;
  }
  return Status::OK;
}

Status DetInferNode::MindXModelInfer(std::shared_ptr<CommonData> data) {
  std::vector<uint32_t> shape;
  uint32_t batchSize = 1;
  shape.push_back(batchSize);
  shape.push_back(CHANNEL_SIZE);
  shape.push_back(data->resizeHeight);
  shape.push_back(data->resizeWidth);
  MxBase::TensorDType tensorDataType = MxBase::TensorDType::FLOAT32;

  MxBase::Tensor imageToTensor(data->imgBuffer, shape, tensorDataType, deviceId_);

  std::vector<MxBase::Tensor> inputs = {};
  inputs.push_back(imageToTensor);

  // (2) 开始推理
  auto inferStartTime = std::chrono::high_resolution_clock::now();
  auto outputs = dbNetMindX_->Infer(inputs);
  auto inferEndTime = std::chrono::high_resolution_clock::now();
  double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
  Profile::detInferTime_ += inferCostTime;
  LogInfo << " [" << data->imgName << "] dbnet infer time cost: " << inferCostTime << "ms.";

  const size_t outputLen = outputs.size();
  if (outputLen <= 0) {
    LogError << "Failed to get model output data";
    return Status::MINDX_MODEL_INFER_FAILURE;
  }
  for (auto &output : outputs) {
    output.ToHost();
  }

  data->outputMindXTensorVec = {outputs[0]};
  data->backendType = BackendType::ACL;
  if (data->imgBuffer != nullptr) {
    delete data->imgBuffer;
    data->imgBuffer = nullptr;
  }
  return Status::OK;
}
