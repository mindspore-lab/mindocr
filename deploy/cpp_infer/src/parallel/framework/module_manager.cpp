#include <utility>
#include "Log/Log.h"
#include "framework/module_manager.h"

namespace AscendBaseModule {
const int MODULE_QUEUE_SIZE = 200;

ModuleManager::ModuleManager() = default;

ModuleManager::~ModuleManager() = default;

Status ModuleManager::Init(CommandParser *options, const std::string &aclConfigPath) {
  LogDebug << "ModuleManager: begin to init.";
  // load and parse config file
  options_ = options;
  // Init pipeline module
  Status ret = InitPipelineModule();
  if (ret != Status::OK) {
    LogError << "ModuleManager: fail to init pipeline module.";
    return ret;
  }
  return ret;
}

Status ModuleManager::InitModuleInstance(const std::shared_ptr<ModuleBase> &moduleInstance, int instanceId,
                                         const std::string &pipelineName, const std::string &moduleName) {
  LogDebug << "ModuleManager: begin to init module instance." << moduleName;
  ModuleInitArgs initArgs;
  initArgs.pipelineName = pipelineName;
  initArgs.moduleName = moduleName;
  initArgs.instanceId = instanceId;

  // Initialize the Init function of each module
  auto backend = Utils::ConvertBackendTypeToEnum(options_->GetStringOption("--backend"));
  if (backend == BackendType::LITE) {
    auto context = std::make_shared<mindspore::Context>();
    if (context == nullptr) {
      LogError << "New mindspore lite context failed.";
      return Status::FAILURE;
    }
    auto &deviceList = context->MutableDeviceInfo();
    auto ascendDeviceInfo = std::make_shared<mindspore::AscendDeviceInfo>();
    if (ascendDeviceInfo == nullptr) {
      LogError << "New AscendDeviceInfo failed";
      return Status::FAILURE;
    }

    std::vector<uint32_t> deviceIdVec;
    options_->GetVectorUint32Value("--device_id", &deviceIdVec);
    ascendDeviceInfo->SetDeviceID(deviceIdVec[0]);
    deviceList.push_back(ascendDeviceInfo);
    initArgs.context = context;
  }
  Status ret = moduleInstance->Init(options_, initArgs);
  if (ret != Status::OK) {
    LogError << "ModuleManager: fail to init module, name = " << moduleName.c_str() << ", instance id = " <<
             instanceId << ".";
    return ret;
  }
  LogDebug << "ModuleManager: module " << initArgs.moduleName << "[" << instanceId << "] init success.";
  return ret;
}

Status ModuleManager::RegisterModules(const std::string &pipelineName, ModuleDesc *modulesDesc, int moduleTypeCount,
                                      int defaultCount) {
  auto iter = pipelineMap_.find(pipelineName);
  std::map<std::string, ModulesInfo> modulesInfoMap;
  if (iter != pipelineMap_.end()) {
    modulesInfoMap = iter->second;
  }

  std::shared_ptr<ModuleBase> moduleInstance = nullptr;

  // create new object of module
  // auto initialize the Init function of each module
  for (int i = 0; i < moduleTypeCount; i++) {
    ModuleDesc moduleDesc = modulesDesc[i];
    int moduleCount = (moduleDesc.moduleCount == -1) ? defaultCount : moduleDesc.moduleCount;
    ModulesInfo modulesInfo;
    for (int j = 0; j < moduleCount; j++) {
      moduleInstance.reset(static_cast<ModuleBase *>(ModuleFactory::MakeModule(moduleDesc.moduleName)));
      Status ret = InitModuleInstance(moduleInstance, j, pipelineName, moduleDesc.moduleName);
      if (ret != Status::OK) {
        return ret;
      }
      modulesInfo.moduleVec.push_back(moduleInstance);
    }
    modulesInfoMap[moduleDesc.moduleName] = modulesInfo;
  }

  pipelineMap_[pipelineName] = modulesInfoMap;

  return Status::OK;
}

Status ModuleManager::RegisterModuleConnects(const std::string &pipelineName, ModuleConnectDesc *connectDescArr,
                                             int moduleConnectCount) {
  auto iter = pipelineMap_.find(pipelineName);
  std::map<std::string, ModulesInfo> modulesInfoMap;
  if (iter == pipelineMap_.end()) {
    return Status::COMM_INVALID_PARAM;
  }
  modulesInfoMap = iter->second;

  std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> dataQueue = nullptr;

  // add connect
  for (int i = 0; i < moduleConnectCount; i++) {
    ModuleConnectDesc connectDesc = connectDescArr[i];
    LogDebug << "Add Connect " << connectDesc.moduleSend << " " << connectDesc.moduleRecv << " type " <<
             connectDesc.connectType;
    auto iterSend = modulesInfoMap.find(connectDesc.moduleSend);
    auto iterRecv = modulesInfoMap.find(connectDesc.moduleRecv);
    if (iterSend == modulesInfoMap.end() || iterRecv == modulesInfoMap.end()) {
      LogError << "Cann't find Module";
      return Status::COMM_INVALID_PARAM;
    }

    ModulesInfo moduleInfoSend = iterSend->second;
    ModulesInfo moduleInfoRecv = iterRecv->second;

    // create input queue for recv module
    for (unsigned int j = 0; j < moduleInfoRecv.moduleVec.size(); j++) {
      dataQueue = std::make_shared<BlockingQueue<std::shared_ptr<void>>>(MODULE_QUEUE_SIZE);
      moduleInfoRecv.inputQueueVec.push_back(dataQueue);
    }
    RegisterInputVec(pipelineName, connectDesc.moduleRecv, moduleInfoRecv.inputQueueVec);

    //
    RegisterOutputModule(pipelineName, connectDesc.moduleSend, connectDesc.moduleRecv, connectDesc.connectType,
                         moduleInfoRecv.inputQueueVec);
  }
  return Status::OK;
}

Status ModuleManager::RegisterInputVec(const std::string &pipelineName, const std::string &moduleName,
                                       std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec) {
  auto pipelineIter = pipelineMap_.find(pipelineName);
  std::map<std::string, ModulesInfo> modulesInfoMap;
  if (pipelineIter == pipelineMap_.end()) {
    return Status::COMM_INVALID_PARAM;
  }
  modulesInfoMap = pipelineIter->second;

  // set inputQueue
  auto iter = modulesInfoMap.find(moduleName);
  if (iter != modulesInfoMap.end()) {
    ModulesInfo moduleInfo = iter->second;
    if (moduleInfo.moduleVec.size() != inputQueVec.size()) {
      return Status::COMM_FAILURE;
    }
    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue = nullptr;
    for (unsigned int j = 0; j < moduleInfo.moduleVec.size(); j++) {
      std::shared_ptr<ModuleBase> moduleInstance = moduleInfo.moduleVec[j];
      inputQueue = inputQueVec[j];
      moduleInstance->SetInputVec(inputQueue);
    }
  }
  return Status::OK;
}

Status
ModuleManager::RegisterOutputModule(const std::string &pipelineName, const std::string &moduleSend,
                                    const std::string &moduleRecv,
                                    ModuleConnectType connectType,
                                    const std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>>
                                    &outputQueVec) {
  auto pipelineIter = pipelineMap_.find(pipelineName);
  std::map<std::string, ModulesInfo> modulesInfoMap;
  if (pipelineIter == pipelineMap_.end()) {
    return Status::COMM_INVALID_PARAM;
  }
  modulesInfoMap = pipelineIter->second;

  // set outputInfo
  auto iter = modulesInfoMap.find(moduleSend);
  if (iter != modulesInfoMap.end()) {
    ModulesInfo moduleInfo = iter->second;
    for (const auto &moduleInstance : moduleInfo.moduleVec) {
      moduleInstance->SetOutputInfo(moduleRecv, connectType, outputQueVec);
    }
  }
  return Status::OK;
}

Status ModuleManager::InitPipelineModule() {
  return Status::OK;
}

Status ModuleManager::RunPipeline() {
  LogInfo << "ModuleManager: begin to run pipeline.";

  // start the thread of the corresponding module
  std::map<std::string, ModulesInfo> modulesInfoMap;
  std::shared_ptr<ModuleBase> moduleInstance;
  for (auto &pipelineIter : pipelineMap_) {
    modulesInfoMap = pipelineIter.second;

    for (auto &iter : modulesInfoMap) {
      ModulesInfo modulesInfo = iter.second;
      for (const auto &i : modulesInfo.moduleVec) {
        moduleInstance = i;
        Status ret = moduleInstance->Run();
        if (ret != Status::OK) {
          LogError << "ModuleManager: fail to run module ";
          return ret;
        }
      }
    }
  }

  LogInfo << "ModuleManager: run pipeline success.";
  return Status::OK;
}

Status ModuleManager::DeInit() {
  LogInfo << "begin to deinit module manager.";
  Status ret = Status::OK;

  // DeInit pipeline module
  ret = DeInitPipelineModule();
  if (ret != Status::OK) {
    LogError << "ModuleManager: fail to deinit pipeline module, ret[%d]" << int(ret) << ".";
    return ret;
  }
  return ret;
}

void ModuleManager::StopModule(const std::shared_ptr<ModuleBase> &module) {
  LogDebug << module->GetModuleName() << "[" << module->GetInstanceId() << "] stop begin";
  Status ret = module->Stop();
  if (ret != Status::OK) {
    LogError << module->GetModuleName() << "[" << module->GetInstanceId() << "] stop failed";
  }
  LogInfo << module->GetModuleName() << "[" << module->GetInstanceId() << "] stop success";
}

Status ModuleManager::DeInitPipelineModule() {
  LogDebug << "begin to deinit pipeline module.";

  std::map<std::string, ModulesInfo> modulesInfoMap;
  for (auto &pipelineIter : pipelineMap_) {
    modulesInfoMap = pipelineIter.second;

    for (auto &iter : modulesInfoMap) {
      ModulesInfo modulesInfo = iter.second;
      std::vector<std::thread> threadVec;

      for (auto &moduleInstance : modulesInfo.moduleVec) {
        threadVec.emplace_back(ModuleManager::StopModule, moduleInstance);
      }

      for (auto &t : threadVec) {
        t.join();
      }
    }
    LogInfo << "Deinit " << pipelineIter.first << " success.";
  }
  return Status::OK;
}
}  // namespace AscendBaseModule
