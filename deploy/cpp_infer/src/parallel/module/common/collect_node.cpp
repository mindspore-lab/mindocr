#include <map>
#include "utils/utils.h"
#include "common/collect_node.h"

const int DBNET_REC_RESULT_SIZE = 9;
const int DBNET_RESULT_SIZE = 8;
const int TEXT_INDEX = 8;
const int FIRST_X_INDEX = 0;
const int FIRST_Y_INDEX = 1;
const int SECOND_X_INDEX = 2;
const int SECOND_Y_INDEX = 3;
const int THIRD_X_INDEX = 4;
const int THIRD_Y_INDEX = 5;
const int FORTH_X_INDEX = 6;
const int FORTH_Y_INDEX = 7;

using AscendBaseModule::ModuleInitArgs;

std::map<TaskType, std::string> RESULTS_SAVE_FILENAME = {{TaskType::DET, "det_results.txt"},
                                                         {TaskType::CLS, "cls_results.txt"},
                                                         {TaskType::REC, "rec_results.txt"},
                                                         {TaskType::DET_REC, "pipeline_results.txt"},
                                                         {TaskType::DET_CLS_REC, "pipeline_results.txt"}};

CollectNode::CollectNode() {
  withoutInputQueue_ = false;
  isStop_ = false;
}

CollectNode::~CollectNode() = default;

Status CollectNode::Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) {
  LogInfo << "Begin to init instance " << initArgs.instanceId;

  AssignInitArgs(initArgs);
  Status ret = ParseConfig(options);
  if (ret != Status::OK) {
    LogError << "collect_process[" << instanceId_ << "]: Fail to parse config params.";
    return ret;
  }

  LogInfo << "collect_process [" << instanceId_ << "] Init success.";
  return Status::OK;
}

Status CollectNode::DeInit() {
  LogInfo << "collect_process [" << instanceId_ << "]: Deinit success.";
  return Status::OK;
}

Status CollectNode::ParseConfig(CommandParser *options) {
  taskType_ = Utils::GetTaskType(options);
  if (taskType_ == TaskType::UNSUPPORTED) {
    LogError << "Unsupported task type";
    return Status::UNSUPPORTED_TASK_TYPE;
  }
  resultPath_ = options->GetStringOption("--res_save_dir");
  if (resultPath_.empty()) {
    return Status::COMM_INVALID_PARAM;
  }
  if (resultPath_[resultPath_.size() - 1] != '/') {
    resultPath_ += "/";
  }
  Utils::MakeDir(resultPath_, false);
  resultPath_ += RESULTS_SAVE_FILENAME[taskType_];
  return Status::OK;
}

void CollectNode::SignalSend(int imgTotal) {
  if (inferSize_ == imgTotal) {
    Profile::GetInstance().GetStoppedThreadNum()++;
    if (Profile::GetInstance().GetStoppedThreadNum() == Profile::GetInstance().GetThreadNum()) {
      Profile::signalReceived_ = true;
    }
  }
}

Status CollectNode::Process(std::shared_ptr<void> commonData) {
  std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
  if (!resultPath_.empty()) {
    std::ofstream outfile(resultPath_, std::ios::out | std::ios::app);
    std::string lineRes;
    if (taskType_ == TaskType::DET_CLS_REC || taskType_ == TaskType::DET_REC) {
      lineRes = GenerateDetClsRecInferRes(data->imageName, data->frameSize, data->inferRes);
    } else if (taskType_ == TaskType::DET) {
      lineRes = GenerateDetInferRes(data->imageName, data->inferRes);
    } else if (taskType_ == TaskType::CLS) {
      lineRes = GenerateClsInferRes(data->imageName, data->inferRes);
    } else if (taskType_ == TaskType::REC) {
      lineRes = GenerateRecInferRes(data->imageName, data->inferRes);
    }
    outfile << lineRes << std::endl;
    outfile.close();
    LogInfo << "----------------------- Save infer result to " << data->imageName << " succeed.";
  }

  auto it = idMap_.find(data->imgId);
  if (it == idMap_.end()) {
    int remaining = data->subImgTotal - data->inferRes.size();
    if (remaining) {
      idMap_.insert({data->imgId, remaining});
    } else {
      inferSize_ += 1;
    }
  } else {
    it->second -= data->inferRes.size();
    if (it->second == 0) {
      idMap_.erase(it);
      inferSize_ += 1;
    }
  }
  SignalSend(data->imgTotal);
  return Status::OK;
}

std::string CollectNode::GenerateDetClsRecInferRes(const std::string &imageName, uint32_t frameSize,
                                                   const std::vector<std::string> &inferRes) {
  std::string result;
  result.append(imageName).append("\t[");
  for (uint32_t i = 0; i < frameSize; ++i) {
    auto resultArr = Utils::SplitString(inferRes[i], ',');
    if (resultArr.size() != DBNET_REC_RESULT_SIZE) {
      continue;
    }
    result.append(R"({"transcription": ")")
        .append(resultArr[TEXT_INDEX])
        .append("\", ")
        .append("\"points\": ")
        .append("[[")
        .append(resultArr[FIRST_X_INDEX])
        .append(", ")
        .append(resultArr[FIRST_Y_INDEX])
        .append("], ")
        .append("[")
        .append(resultArr[SECOND_X_INDEX])
        .append(", ")
        .append(resultArr[SECOND_Y_INDEX])
        .append("], ")
        .append("[")
        .append(resultArr[THIRD_X_INDEX])
        .append(", ")
        .append(resultArr[THIRD_Y_INDEX])
        .append("], ")
        .append("[")
        .append(resultArr[FORTH_X_INDEX])
        .append(", ")
        .append(resultArr[FORTH_Y_INDEX])
        .append("]]}");
    if (i != frameSize - 1) {
      result.append(", ");
    }
  }
  result.append("]");
  return result;
}

std::string
CollectNode::GenerateDetInferRes(const std::string &imageName, const std::vector<std::string> &inferRes) {
  std::string result;
  result.append(imageName).append("\t[");
  for (size_t i = 0; i < inferRes.size(); i++) {
    auto resultArr = Utils::SplitString(inferRes[i], ',');
    if (resultArr.size() != DBNET_RESULT_SIZE) {
      continue;
    }
    result.append("[[")
        .append(resultArr[FIRST_X_INDEX])
        .append(", ")
        .append(resultArr[FIRST_Y_INDEX])
        .append("], ")
        .append("[")
        .append(resultArr[SECOND_X_INDEX])
        .append(", ")
        .append(resultArr[SECOND_Y_INDEX])
        .append("], ")
        .append("[")
        .append(resultArr[THIRD_X_INDEX])
        .append(", ")
        .append(resultArr[THIRD_Y_INDEX])
        .append("], ")
        .append("[")
        .append(resultArr[FORTH_X_INDEX])
        .append(", ")
        .append(resultArr[FORTH_Y_INDEX])
        .append("]]");
    if (i != inferRes.size() - 1) {
      result.append(",");
    }
  }
  result.append("]");
  return result;
}

std::string
CollectNode::GenerateClsInferRes(const std::string &imageName, const std::vector<std::string> &inferRes) {
  std::string result;
  result.append(imageName).append("\t");
  for (const auto &s : inferRes) {
    result.append(s).append(" ");
  }
  return result;
}

std::string
CollectNode::GenerateRecInferRes(const std::string &imageName, const std::vector<std::string> &inferRes) {
  std::string result;
  result.append(imageName).append("\t");
  for (const auto &s : inferRes) {
    result.append(s).append(" ");
  }
  return result;
}
