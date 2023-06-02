#include <chrono>
#include <utility>
#include <memory>
#include <string>
#include "Log/Log.h"
#include "blocking_queue/blocking_queue.h"
#include "framework/module_base.h"

namespace AscendBaseModule {
const int INPUT_QUEUE_WARN_SIZE = 32;
const double TIME_COUNTS = 1000.0;

void ModuleBase::AssignInitArgs(const ModuleInitArgs &initArgs) {
  pipelineName_ = initArgs.pipelineName;
  moduleName_ = initArgs.moduleName;
  instanceId_ = initArgs.instanceId;
  isStop_ = false;
}

// run module instance in a new thread created
Status ModuleBase::Run() {
  LogDebug << moduleName_ << "[" << instanceId_ << "] Run";
  processThr_ = std::thread(&ModuleBase::ProcessThread, this);
  return Status::OK;
}

// get the data from input queue then call Process function in the new thread
void ModuleBase::ProcessThread() {
  Status ret;
  // if the module has no input queue, call Process function directly.
  if (withoutInputQueue_) {
    ret = Process(nullptr);
    if (ret != Status::OK) {
      LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]";
    }
    return;
  }
  if (inputQueue_ == nullptr) {
    LogError << "Invalid input queue of " << moduleName_ << "[" << instanceId_ << "].";
    return;
  }
  LogDebug << "Input queue for " << moduleName_ << "[" << instanceId_ << "], inputQueue=" << inputQueue_;
  // repeatly pop data from input queue and call the Process funtion. Results will be pushed to output queues.
  while (!isStop_) {
    std::shared_ptr<void> frameInfo = nullptr;
    ret = inputQueue_->Pop(&frameInfo);
    if (ret == Status::QUEUE_STOPPED) {
      LogDebug << moduleName_ << "[" << instanceId_ << "] input queue Stopped";
      break;
    } else if (ret != Status::OK || frameInfo == nullptr) {
      LogError << "Fail to get data from input queue for " << moduleName_ << "[" << instanceId_ << "]";
      continue;
    }
    CallProcess(frameInfo);
  }
  LogInfo << moduleName_ << "[" << instanceId_ << "] process thread End";
}

void ModuleBase::CallProcess(const std::shared_ptr<void> &sendData) {
  auto startTime = std::chrono::high_resolution_clock::now();
  Status ret = Status::COMM_FAILURE;
  try {
    ret = Process(sendData);
  } catch (...) {
    LogError << "error occurred in " << moduleName_ << "[" << instanceId_ << "]";
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  int queueSize = inputQueue_->GetSize();
  if (queueSize > INPUT_QUEUE_WARN_SIZE) {
    LogWarn << "[Statistic] [Module] [" << moduleName_ << "] [" << instanceId_ << "] [QueueSize] [" << queueSize
            << "] [Process] [" << costMs << " ms]";
  }

  if (ret != Status::OK) {
    LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]";
  }
}

void ModuleBase::SetOutputInfo(std::string moduleName, ModuleConnectType connectType,
                               std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec) {
  if (outputQueVec.size() == 0) {
    LogError << "outputQueVec is Empty! " << moduleName;
    return;
  }

  ModuleOutputInfo outputInfo;
  outputInfo.moduleName = moduleName;
  outputInfo.connectType = connectType;
  outputInfo.outputQueVec = outputQueVec;
  outputInfo.outputQueVecSize = outputQueVec.size();
  outputQueMap_[moduleName] = outputInfo;
}

std::string ModuleBase::GetModuleName() {
  return moduleName_;
}

int ModuleBase::GetInstanceId() {
  return instanceId_;
}

void ModuleBase::SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue) {
  inputQueue_ = std::move(inputQueue);
}

void ModuleBase::SendToNextModule(const std::string &moduleName,
                                  const std::shared_ptr<void> &outputData,
                                  int channelId) {
  if (isStop_) {
    LogDebug << moduleName_ << "[" << instanceId_ << "] is Stopped, can't send to next module";
    return;
  }

  if (outputQueMap_.find(moduleName) == outputQueMap_.end()) {
    LogError << "No Next Module " << moduleName;
    return;
  }

  auto itr = outputQueMap_.find(moduleName);
  if (itr == outputQueMap_.end()) {
    LogError << "No Next Module " << moduleName;
    return;
  }
  ModuleOutputInfo outputInfo = itr->second;

  if (outputInfo.connectType == MODULE_CONNECT_ONE) {
    outputInfo.outputQueVec[0]->Push(outputData, true);
  } else if (outputInfo.connectType == MODULE_CONNECT_CHANNEL) {
    uint32_t ch = channelId % outputInfo.outputQueVecSize;
    if (ch >= outputInfo.outputQueVecSize) {
      LogError << "No Next Module!";
      return;
    }
    outputInfo.outputQueVec[ch]->Push(outputData, true);
  } else if (outputInfo.connectType == MODULE_CONNECT_PAIR) {
    outputInfo.outputQueVec[instanceId_]->Push(outputData, true);
  } else if (outputInfo.connectType == MODULE_CONNECT_RANDOM) {
    outputInfo.outputQueVec[sendCount_ % outputInfo.outputQueVecSize]->Push(outputData, true);
  }
  sendCount_++;
}

// clear input queue and stop the thread of the instance, called before destroy the instance
Status ModuleBase::Stop() {
  // stop input queue
  isStop_ = true;

  if (inputQueue_ != nullptr) {
    inputQueue_->Stop();
  }

  if (processThr_.joinable()) {
    processThr_.join();
  }

  return DeInit();
}
}  // namespace AscendBaseModule
