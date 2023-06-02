#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_FRAMEWORK_MODULE_MANAGER_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_FRAMEWORK_MODULE_MANAGER_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include "Log/Log.h"
#include "framework/module_base.h"
#include "framework/module_factory.h"
#include "command_parser/command_parser.h"
#include "utils/utils.h"

namespace AscendBaseModule {

struct ModuleDesc {
  std::string moduleName;
  int moduleCount;  // -1 using the defaultCount
};

struct ModuleConnectDesc {
  std::string moduleSend;
  std::string moduleRecv;
  ModuleConnectType connectType;
};

// information for one type of module
struct ModulesInformation {
  std::vector<std::shared_ptr<ModuleBase>> moduleVec;
  std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueueVec;
};

using ModulesInfo = ModulesInformation;

class ModuleManager {
 public:
  ModuleManager();

  ~ModuleManager();

  Status Init(CommandParser *options, const std::string &aclConfigPath);

  Status DeInit();

  Status
  RegisterModules(const std::string &pipelineName, ModuleDesc *moduleDesc, int moduleTypeCount, int defaultCount);

  Status
  RegisterModuleConnects(const std::string &pipelineName, ModuleConnectDesc *connectDescArr,
                         int moduleConnectCount);

  Status RegisterInputVec(const std::string &pipelineName, const std::string &moduleName,
                          std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec);

  Status RegisterOutputModule(const std::string &pipelineName, const std::string &moduleSend,
                              const std::string &moduleRecv,
                              ModuleConnectType connectType,
                              const std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &outputQueVec);

  Status RunPipeline();

 private:
  Status InitModuleInstance(const std::shared_ptr<ModuleBase> &moduleInstance, int instanceId,
                            const std::string &pipelineName, const std::string &moduleName);

  Status InitPipelineModule();

  Status DeInitPipelineModule();

  static void StopModule(const std::shared_ptr<ModuleBase> &moduleInstance);

 private:
  int32_t deviceId_ = 0;
  std::map<std::string, std::map<std::string, ModulesInfo>> pipelineMap_ = {};
  ConfigParser configParser_ = {};
  CommandParser* options_ = {};
};
}  // namespace AscendBaseModule

#endif
