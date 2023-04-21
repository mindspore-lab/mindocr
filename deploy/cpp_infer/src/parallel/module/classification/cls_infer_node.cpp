#include <string>
#include <memory>
#include "data_type/constant.h"
#include "classification/cls_infer_node.h"
#include "classification/cls_post_node.h"

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_ClsPostNode;

ClsInferNode::ClsInferNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

ClsInferNode::~ClsInferNode() = default;

Status ClsInferNode::Init(CommandParser *options, const ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;
  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options, initArgs);
  if (ret != Status::OK) {
    LogError << "cls_infer_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }
  LogInfo << "cls_infer_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status ClsInferNode::DeInit() {
  LogInfo << "cls_infer_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status ClsInferNode::ParseConfig(CommandParser *options, const ModuleInitArgs &initArgs) {
  std::vector<uint32_t> deviceIdVec;
  Status ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK || deviceIdVec.empty()) {
    LogError << "Get device id failed, please check the value of deviceId";
    return Status::COMM_INVALID_PARAM;
  }
  deviceId_ = (int32_t) deviceIdVec[instanceId_ % deviceIdVec.size()];
  LogDebug << "deviceId: " << deviceId_;

  std::string clsModelPath = options->GetStringOption("--cls_model_path");
  LogDebug << "clsModelPath: " << clsModelPath;

  ret = Utils::CheckPath(clsModelPath, "clsModelPath");
  if (ret != Status::OK) {
    LogError << "rec model path: " << clsModelPath << " is not exist of can not read.";
    return Status::COMM_INVALID_PARAM;
  }
  backend_ = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  if (backend_ == BackendType::ACL) {
    clsNetMindX_ = std::make_unique<MxBase::Model>(clsModelPath, deviceId_);
  } else if (backend_ == BackendType::LITE) {
    clsNetLite_ = new mindspore::Model();
    auto buildRet = clsNetLite_->Build(clsModelPath, mindspore::kMindIR, initArgs.context);
    if (buildRet != mindspore::kSuccess) {
      LogError << "Build lite model error" << buildRet;
      return Status::LITE_MODEL_BUILD_FAILURE;
    }
  } else {
    return Status::UNSUPPORTED_INFER_ENGINE;
  }
  clsHeight_ = DEFAULT_CLS_HEIGHT;
  clsWidth_ = DEFAULT_CLS_WIDTH;
  return Status::OK;
}

Status ClsInferNode::LiteModelInfer(const std::shared_ptr<CommonData> &data) const {
  std::vector<int64_t> shape;
  shape.push_back(data->batchSize);
  shape.push_back(CHANNEL_SIZE);
  shape.push_back(clsHeight_);
  shape.push_back(clsWidth_);
  auto inputs = clsNetLite_->GetInputs();
  std::vector<std::vector<int64_t>> dims;
  dims.push_back(shape);
  clsNetLite_->Resize(inputs, dims);
  inputs[0].SetData(data->imgBuffer);

  std::vector<mindspore::MSTensor> outputs;

  auto inferStartTime = std::chrono::high_resolution_clock::now();
  auto predictResult = clsNetLite_->Predict(inputs, &outputs);
  if (predictResult != mindspore::kSuccess) {
    LogError << "Lite predict error";
    return Status::LITE_MODEL_INFER_FAILURE;
  }
  auto inferEndTime = std::chrono::high_resolution_clock::now();
  double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
  Profile::clsInferTime_ += inferCostTime;
  data->outputLiteTensorVec = outputs;
  return Status::OK;
}

Status ClsInferNode::MindXModelInfer(const std::shared_ptr<CommonData> &data) const {
  std::vector<uint32_t> shape;
  shape.push_back(data->batchSize);
  shape.push_back(CHANNEL_SIZE);
  shape.push_back(clsHeight_);
  shape.push_back(clsWidth_);
  MxBase::TensorDType tensorDataType = MxBase::TensorDType::FLOAT32;
  std::vector<MxBase::Tensor> inputs = {};

  MxBase::Tensor imageToTensor(data->imgBuffer, shape, tensorDataType, deviceId_);
  inputs.push_back(imageToTensor);

  auto inferStartTime = std::chrono::high_resolution_clock::now();
  auto outputs = clsNetMindX_->Infer(inputs);
  auto inferEndTime = std::chrono::high_resolution_clock::now();
  double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
  Profile::clsInferTime_ += inferCostTime;
  for (auto &output : outputs) {
    output.ToHost();
  }
  data->outputMindXTensorVec = outputs;
  return Status::OK;
}

Status ClsInferNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  if (data->eof) {
    SendToNextModule(MT_ClsPostNode, data, data->channelId);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Profile::clsInferProcessTime_ += costTime;
    Profile::e2eProcessTime_ += costTime;
    return Status::OK;
  }
  data->backendType = backend_;

  if (backend_ == BackendType::ACL) {
    auto ret = MindXModelInfer(data);
    if (ret != Status::OK) {
      return ret;
    }
  } else if (backend_ == BackendType::LITE) {
    auto ret = LiteModelInfer(data);
    if (ret != Status::OK) {
      return ret;
    }
  } else {
    return Status::UNSUPPORTED_INFER_ENGINE;
  }
  LogDebug << "cls_infer_process end.";
  if (data->imgBuffer != nullptr) {
    delete data->imgBuffer;
    data->imgBuffer = nullptr;
  }
  SendToNextModule(MT_ClsPostNode, data, data->channelId);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  Profile::clsInferProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  return Status::OK;
}
