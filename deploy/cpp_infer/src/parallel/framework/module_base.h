#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_FRAMEWORK_MODULE_BASE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_FRAMEWORK_MODULE_BASE_H_

#include <thread>
#include <vector>
#include <map>
#include <atomic>
#include <memory>
#include <string>
#include "include/api/context.h"
#include "config_parser/config_parser.h"
#include "blocking_queue/blocking_queue.h"
#include "command_parser/command_parser.h"

namespace AscendBaseModule {
enum ModuleConnectType {
  MODULE_CONNECT_ONE = 0,
  MODULE_CONNECT_CHANNEL,
  MODULE_CONNECT_PAIR,
  MODULE_CONNECT_RANDOM
};

struct ModuleInitArguments {
  std::string pipelineName = {};
  std::string moduleName = {};
  int instanceId = -1;
  std::shared_ptr<mindspore::Context> context;
};

struct ModuleOutputInformation {
  std::string moduleName;
  ModuleConnectType connectType = MODULE_CONNECT_RANDOM;
  std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec = {};
  uint32_t outputQueVecSize = 0;
};

using ModuleInitArgs = ModuleInitArguments;
using ModuleOutputInfo = ModuleOutputInformation;

class ModuleBase {
 public:
  ModuleBase() = default;

  virtual ~ModuleBase() = default;

  virtual Status Init(CommandParser *options, const ModuleInitArgs &initArgs) = 0;

  virtual Status DeInit() = 0;

  Status Run();
  Status Stop();

  void SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue);

  void SetOutputInfo(std::string moduleName, ModuleConnectType connectType,
                     std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);

  void SendToNextModule(const std::string &moduleNext, const std::shared_ptr<void> &outputData, int channelId = 0);

  std::string GetModuleName();

  int GetInstanceId();

 protected:
  void ProcessThread();

  virtual Status Process(std::shared_ptr<void> inputData) = 0;

  void CallProcess(const std::shared_ptr<void> &sendData);

  void AssignInitArgs(const ModuleInitArgs &initArgs);

 protected:
  int instanceId_ = -1;
  std::string pipelineName_ = {};
  std::string moduleName_ = {};
  int32_t deviceId_ = -1;
  std::thread processThr_ = {};
  std::atomic_bool isStop_ = {};
  bool withoutInputQueue_ = false;
  std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue_ = nullptr;
  std::map<std::string, ModuleOutputInfo> outputQueMap_ = {};
  int sendCount_ = 0;
};
}  // namespace AscendBaseModule
#endif
