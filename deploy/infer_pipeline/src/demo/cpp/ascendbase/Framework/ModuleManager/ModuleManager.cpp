/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ModuleManager/ModuleManager.h"
#include "Log/Log.h"
#ifdef ASCEND_MODULE_USE_ACL
#include "ResourceManager/ResourceManager.h"
#endif

namespace ascendBaseModule {
const int MODULE_QUEUE_SIZE = 200;

ModuleManager::ModuleManager() {}

ModuleManager::~ModuleManager() {}

APP_ERROR ModuleManager::Init(std::string &configPath, std::string &aclConfigPath)
{
    LogDebug << "ModuleManager: begin to init.";

    // load and parse config file
    APP_ERROR ret = configParser_.ParseConfig(configPath);
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: cannot parse file.";
        return ret;
    }

    // Init Acl
#ifdef ASCEND_MODULE_USE_ACL
    ret = InitAcl(aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: fail to init Acl.";
        return ret;
    }
#endif

    // Init pipeline module
    ret = InitPipelineModule();
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: fail to init pipeline module.";
        return ret;
    }

    return ret;
}

#ifdef ASCEND_MODULE_USE_ACL
APP_ERROR ModuleManager::InitAcl(std::string &aclConfigPath)
{
    LogDebug << "ModuleManager: begin to init Acl.";
    std::string itemCfgStr;

    itemCfgStr = "SystemConfig.deviceId";
    APP_ERROR ret = configParser_.GetIntValue(itemCfgStr, deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: fail to get device id.";
        return ret;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: fail to get run mode of device, ret=" << ret << ".";
        return ret;
    }

    if (runMode_ == ACL_DEVICE) {
        deviceId_ = 0;
    } else if (deviceId_ < 0) {
        LogError << "ModuleManager: invalid device id, deviceId=" << deviceId_ << ".";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ResourceInfo resourceInfo;
    resourceInfo.aclConfigPath = aclConfigPath;
    resourceInfo.deviceIds.insert(deviceId_);
    return ResourceManager::GetInstance()->InitResource(resourceInfo);
}
#endif

APP_ERROR ModuleManager::InitModuleInstance(std::shared_ptr<ModuleBase> moduleInstance, int instanceId,
    std::string pipelineName, std::string moduleName)
{
    LogDebug << "ModuleManager: begin to init module instance." << moduleName;
    ModuleInitArgs initArgs;
#ifdef ASCEND_MODULE_USE_ACL
    initArgs.context = ResourceManager::GetInstance()->GetContext(deviceId_);
    initArgs.runMode = runMode_;
#endif
    initArgs.pipelineName = pipelineName;
    initArgs.moduleName = moduleName;
    initArgs.instanceId = instanceId;

    // Initialize the Init function of each module
    APP_ERROR ret = moduleInstance->Init(configParser_, initArgs);
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: fail to init module, name = " << moduleName.c_str() << ", instance id = " <<
            instanceId << ".";
        return ret;
    }
    LogDebug << "ModuleManager: module " << initArgs.moduleName << "[" << instanceId << "] init success.";
    return ret;
}

APP_ERROR ModuleManager::RegisterModules(std::string pipelineName, ModuleDesc *modulesDesc, int moduleTypeCount,
    int defaultCount)
{
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
            APP_ERROR ret = InitModuleInstance(moduleInstance, j, pipelineName, moduleDesc.moduleName);
            if (ret != APP_ERR_OK) {
                return ret;
            }
            modulesInfo.moduleVec.push_back(moduleInstance);
        }
        modulesInfoMap[moduleDesc.moduleName] = modulesInfo;
    }

    pipelineMap_[pipelineName] = modulesInfoMap;

    return APP_ERR_OK;
}

APP_ERROR ModuleManager::RegisterModuleConnects(std::string pipelineName, ModuleConnectDesc *connnectDesc,
    int moduleConnectCount)
{
    auto iter = pipelineMap_.find(pipelineName);
    std::map<std::string, ModulesInfo> modulesInfoMap;
    if (iter == pipelineMap_.end()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modulesInfoMap = iter->second;

    std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> dataQueue = nullptr;

    // add connect
    for (int i = 0; i < moduleConnectCount; i++) {
        ModuleConnectDesc connectDesc = connnectDesc[i];
        LogDebug << "Add Connect " << connectDesc.moduleSend << " " << connectDesc.moduleRecv << " type " <<
            connectDesc.connectType;
        auto iterSend = modulesInfoMap.find(connectDesc.moduleSend);
        auto iterRecv = modulesInfoMap.find(connectDesc.moduleRecv);
        if (iterSend == modulesInfoMap.end() || iterRecv == modulesInfoMap.end()) {
            LogError << "Cann't find Module";
            return APP_ERR_COMM_INVALID_PARAM;
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
    return APP_ERR_OK;
}

APP_ERROR ModuleManager::RegisterInputVec(std::string pipelineName, std::string moduleName,
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec)
{
    auto pipelineIter = pipelineMap_.find(pipelineName);
    std::map<std::string, ModulesInfo> modulesInfoMap;
    if (pipelineIter == pipelineMap_.end()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modulesInfoMap = pipelineIter->second;

    // set inputQueue
    auto iter = modulesInfoMap.find(moduleName);
    if (iter != modulesInfoMap.end()) {
        ModulesInfo moduleInfo = iter->second;
        if (moduleInfo.moduleVec.size() != inputQueVec.size()) {
            return APP_ERR_COMM_FAILURE;
        }
        std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue = nullptr;
        for (unsigned int j = 0; j < moduleInfo.moduleVec.size(); j++) {
            std::shared_ptr<ModuleBase> moduleInstance = moduleInfo.moduleVec[j];
            inputQueue = inputQueVec[j];
            moduleInstance->SetInputVec(inputQueue);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ModuleManager::RegisterOutputModule(std::string pipelineName, std::string moduleSend, std::string moduleRecv,
    ModuleConnectType connectType, std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec)
{
    auto pipelineIter = pipelineMap_.find(pipelineName);
    std::map<std::string, ModulesInfo> modulesInfoMap;
    if (pipelineIter == pipelineMap_.end()) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    modulesInfoMap = pipelineIter->second;

    // set outputInfo
    auto iter = modulesInfoMap.find(moduleSend);
    if (iter != modulesInfoMap.end()) {
        ModulesInfo moduleInfo = iter->second;
        for (unsigned int j = 0; j < moduleInfo.moduleVec.size(); j++) {
            std::shared_ptr<ModuleBase> moduleInstance = moduleInfo.moduleVec[j];
            moduleInstance->SetOutputInfo(moduleRecv, connectType, outputQueVec);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ModuleManager::InitPipelineModule()
{
    return APP_ERR_OK;
}

APP_ERROR ModuleManager::RunPipeline()
{
    LogInfo << "ModuleManager: begin to run pipeline.";

    // start the thread of the corresponding module
    std::map<std::string, ModulesInfo> modulesInfoMap;
    std::shared_ptr<ModuleBase> moduleInstance;
    for (auto pipelineIter = pipelineMap_.begin(); pipelineIter != pipelineMap_.end(); pipelineIter++) {
        modulesInfoMap = pipelineIter->second;

        for (auto iter = modulesInfoMap.begin(); iter != modulesInfoMap.end(); iter++) {
            ModulesInfo modulesInfo = iter->second;
            for (uint32_t i = 0; i < modulesInfo.moduleVec.size(); i++) {
                moduleInstance = modulesInfo.moduleVec[i];
                APP_ERROR ret = moduleInstance->Run();
                if (ret != APP_ERR_OK) {
                    LogError << "ModuleManager: fail to run module ";
                    return ret;
                }
            }
        }
    }

    LogInfo << "ModuleManager: run pipeline success.";
    return APP_ERR_OK;
}

APP_ERROR ModuleManager::DeInit(void)
{
    LogInfo << "begin to deinit module manager.";
    APP_ERROR ret = APP_ERR_OK;

    // DeInit pipeline module
    ret = DeInitPipelineModule();
    if (ret != APP_ERR_OK) {
        LogError << "ModuleManager: fail to deinit pipeline module, ret[%d]" << ret << ".";
        return ret;
    }

#ifdef ASCEND_MODULE_USE_ACL
    ResourceManager::GetInstance()->Release();
#endif

    return ret;
}

void ModuleManager::StopModule(std::shared_ptr<ModuleBase> module)
{
    LogDebug << module->GetModuleName() << "[" << module->GetInstanceId() << "] stop begin";
    APP_ERROR ret = module->Stop();
    if (ret != APP_ERR_OK) {
        LogError << module->GetModuleName() << "[" << module->GetInstanceId() << "] stop failed";
    }
    LogInfo << module->GetModuleName() << "[" << module->GetInstanceId() << "] stop success";
}

APP_ERROR ModuleManager::DeInitPipelineModule()
{
    LogDebug << "begin to deinit pipeline module.";

    std::map<std::string, ModulesInfo> modulesInfoMap;
    for (auto pipelineIter = pipelineMap_.begin(); pipelineIter != pipelineMap_.end(); pipelineIter++) {
        modulesInfoMap = pipelineIter->second;

        for (auto iter = modulesInfoMap.begin(); iter != modulesInfoMap.end(); iter++) {
            ModulesInfo modulesInfo = iter->second;
            std::vector<std::thread> threadVec;

            for (auto &moduleInstance : modulesInfo.moduleVec) {
                threadVec.emplace_back(ModuleManager::StopModule, moduleInstance);
            }

            for (auto &t : threadVec) {
                t.join();
            }
        }
        LogInfo << "Deinit " << pipelineIter->first << " success.";
    }
    return APP_ERR_OK;
}
}
