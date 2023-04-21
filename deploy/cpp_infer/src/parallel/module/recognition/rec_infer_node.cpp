#include <string>
#include <algorithm>
#include "recognition/rec_post_node.h"
#include "utils/utils.h"
#include "data_type/constant.h"
#include "recognition/rec_infer_node.h"

const int HEIGHT_INDEX = 2;

using AscendBaseModule::ModuleInitArgs;
using AscendBaseModule::MT_RecPostNode;

RecInferNode::RecInferNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

RecInferNode::~RecInferNode() {}

Status RecInferNode::Init(CommandParser *options, const ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;
  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options, initArgs);
  if (ret != Status::OK) {
    LogError << "crnn_infer_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }
  LogInfo << "crnn_infer_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status RecInferNode::DeInit() {
  LogInfo << "crnn_infer_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status RecInferNode::ParseConfig(CommandParser *options, const ModuleInitArgs &initArgs) {
  std::vector<uint32_t> deviceIdVec;
  Status ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK || deviceIdVec.empty()) {
    LogError << "Get device id failed, please check the value of deviceId";
    return Status::COMM_INVALID_PARAM;
  }
  deviceId_ = (int32_t) deviceIdVec[instanceId_ % deviceIdVec.size()];
  backend_ = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  staticMethod_ = options->GetBoolOption("--static_rec_model_mode");
  LogDebug << "staticMethod: " << staticMethod_;

  stdHeight_ = options->GetIntOption("--rec_height");
  LogDebug << "mStdHeight: " << stdHeight_;

  std::string pathExpr = options->GetStringOption("--rec_model_path");
  if (ret != Status::OK) {
    LogError << "Get recModelPath failed, please check the value of recModelPath";
    return Status::COMM_INVALID_PARAM;
  }
  LogDebug << "recModelPath: " << pathExpr;

  ret = Utils::CheckPath(pathExpr, "recModelPath");
  if (ret != Status::OK) {
    LogError << "rec model path: " << pathExpr << " is not exist of can not read.";
    return Status::COMM_INVALID_PARAM;
  }
  if (staticMethod_) {
    std::vector<std::string> files;
    Utils::GetAllFiles(pathExpr, &files);
    std::vector<std::vector<uint64_t>> dynamicGearInfo;
    for (auto &file : files) {
      if (backend_ == BackendType::ACL) {
        crnnNetMindX_.push_back(new MxBase::Model(file, deviceId_));
        dynamicGearInfo = crnnNetMindX_[crnnNetMindX_.size() - 1]->GetDynamicGearInfo();
        stdHeight_ = dynamicGearInfo[0][HEIGHT_INDEX];
        batchSizeList_.push_back(dynamicGearInfo[0][0]);
      } else if (backend_ == BackendType::LITE && Utils::EndsWith(file, ".mindir")) {
        auto model = new mindspore::Model();
        auto buildRet = model->Build(file, mindspore::kMindIR, initArgs.context);
        if (buildRet != mindspore::kSuccess) {
          LogError << "Build lite model error" << buildRet;
          return Status::LITE_MODEL_BUILD_FAILURE;
        }
        dynamicGearInfo = Utils::GetGearInfo(file);
        auto liteModelWrap = new LiteModelWrap;
        liteModelWrap->model = model;
        liteModelWrap->dynamicGearInfo = dynamicGearInfo;
        crnnNetLite_.push_back(liteModelWrap);
        dynamicGearInfo = crnnNetLite_[crnnNetLite_.size() - 1]->dynamicGearInfo;
        stdHeight_ = dynamicGearInfo[0][HEIGHT_INDEX];
        batchSizeList_.push_back(dynamicGearInfo[0][0]);
      }
    }
    std::sort(batchSizeList_.begin(), batchSizeList_.end(), Utils::UintCompare);
    std::sort(crnnNetMindX_.begin(), crnnNetMindX_.end(), Utils::ModelCompare);
    std::sort(crnnNetLite_.begin(), crnnNetLite_.end(), Utils::LiteModelCompare);
  } else {
    if (backend_ == BackendType::ACL) {
      crnnNetMindX_.push_back(new MxBase::Model(pathExpr, deviceId_));
    } else if (backend_ == BackendType::LITE) {
      auto model = new mindspore::Model();
      auto buildRet = model->Build(pathExpr, mindspore::kMindIR, initArgs.context);
      if (buildRet != mindspore::kSuccess) {
        LogError << "Build lite model error " << buildRet;
        return Status::LITE_MODEL_BUILD_FAILURE;
      }
      auto wrapModel = new LiteModelWrap;
      wrapModel->model = model;
      crnnNetLite_.push_back(wrapModel);
    } else {
      return Status::UNSUPPORTED_INFER_ENGINE;
    }
  }

  return Status::OK;
}

Status RecInferNode::MindXModelInfer(const std::shared_ptr<CommonData> &data) {
  auto srcData = data->imgBuffer;
  auto batchSize = data->batchSize;
  auto maxResizedW = data->maxResizedW;

  int modelIndex = 0;
  if (staticMethod_) {
    auto it = find(batchSizeList_.begin(), batchSizeList_.end(), batchSize);
    modelIndex = it - batchSizeList_.begin();
  }

  std::vector<uint32_t> shape;
  shape.push_back(batchSize);
  shape.push_back(CHANNEL_SIZE);
  shape.push_back(stdHeight_);
  shape.push_back(maxResizedW);
  MxBase::TensorDType tensorDataType = MxBase::TensorDType::FLOAT32;

  std::vector<MxBase::Tensor> inputs = {};
  MxBase::Tensor imageToTensor(srcData, shape, tensorDataType, deviceId_);
  inputs.push_back(imageToTensor);

  // (2) 开始推理
  auto inferStartTime = std::chrono::high_resolution_clock::now();
  auto outputs = crnnNetMindX_[modelIndex]->Infer(inputs);
  auto inferEndTime = std::chrono::high_resolution_clock::now();
  double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
  Profile::recInferTime_ += inferCostTime;
  for (auto &output : outputs) {
    output.ToHost();
  }
  data->outputMindXTensorVec = outputs;
  LogInfo << "End Crnn Model Infer progress...";
  return Status::OK;
}

Status RecInferNode::LiteModelInfer(const std::shared_ptr<CommonData> &data) {
  auto srcData = data->imgBuffer;
  auto batchSize = data->batchSize;
  auto maxResizedW = data->maxResizedW;

  int modelIndex = 0;
  if (staticMethod_) {
    auto it = find(batchSizeList_.begin(), batchSizeList_.end(), batchSize);
    modelIndex = it - batchSizeList_.begin();
  }

  std::vector<int64_t> shape;
  shape.push_back(batchSize);
  shape.push_back(CHANNEL_SIZE);
  shape.push_back(stdHeight_);
  shape.push_back(maxResizedW);

  auto model = crnnNetLite_[modelIndex]->model;
  auto inputs = model->GetInputs();

  std::vector<std::vector<int64_t>> dims;
  dims.push_back(shape);

  model->Resize(inputs, dims);
  inputs[0].SetData(srcData);
  // (2) 开始推理
  auto inferStartTime = std::chrono::high_resolution_clock::now();

  std::vector<mindspore::MSTensor> outputs;
  auto predictResult = model->Predict(inputs, &outputs);
  if (predictResult != mindspore::kSuccess) {
    LogError << "Lite predict error";
    return Status::LITE_MODEL_INFER_FAILURE;
  }
  auto inferEndTime = std::chrono::high_resolution_clock::now();
  double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
  Profile::recInferTime_ += inferCostTime;

  data->outputLiteTensorVec = outputs;
  LogInfo << "End crnn model infer progress...";
  return Status::OK;
}

Status RecInferNode::Process(std::shared_ptr<void> commonData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  if (data->eof) {
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Profile::recInferProcessTime_ += costTime;
    Profile::e2eProcessTime_ += costTime;
    SendToNextModule(MT_RecPostNode, data, data->channelId);
    return Status::OK;
  }

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
  auto endTime = std::chrono::high_resolution_clock::now();
  double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  if (data->imgBuffer != nullptr) {
    delete data->imgBuffer;
    data->imgBuffer = nullptr;
  }

  Profile::recInferProcessTime_ += costTime;
  Profile::e2eProcessTime_ += costTime;
  SendToNextModule(MT_RecPostNode, data, data->channelId);

  return Status::OK;
}
