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

#ifndef INC_MODULE_MANAGER_H
#define INC_MODULE_MANAGER_H

#include "acl/acl.h"
#include "Log/Log.h"
#include "ModuleManager/ModuleBase.h"
#include "ModuleManager/ModuleFactory.h"

namespace ascendBaseModule {
const std::string PIPELINE_DEFAULT = "DefaultPipeline";

struct ModuleDesc {
    std::string moduleName;
    int moduleCount; // -1 using the defaultCount
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
    APP_ERROR Init(std::string &configPath, std::string &aclConfigPath);
    APP_ERROR DeInit(void);

    APP_ERROR RegisterModules(std::string pipelineName, ModuleDesc *moduleDesc, int moduleTypeCount, int defaultCount);
    APP_ERROR RegisterModuleConnects(std::string pipelineName, ModuleConnectDesc *connnectDesc, int moduleConnectCount);

    APP_ERROR RegisterInputVec(std::string pipelineName, std::string moduleName,
        std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec);
    APP_ERROR RegisterOutputModule(std::string pipelineName, std::string moduleSend, std::string moduleRecv,
        ModuleConnectType connectType, std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);

    APP_ERROR RunPipeline();

private:
#ifdef ASCEND_MODULE_USE_ACL
    APP_ERROR InitAcl(std::string &aclConfigPath);
#endif
    APP_ERROR InitModuleInstance(std::shared_ptr<ModuleBase> moduleInstance, int instanceId, std::string pipelineName,
        std::string moduleName);
    APP_ERROR InitPipelineModule();
    APP_ERROR DeInitPipelineModule();
    static void StopModule(std::shared_ptr<ModuleBase> moduleInstance);

private:
    int32_t deviceId_ = 0;
#ifdef ASCEND_MODULE_USE_ACL
    aclrtContext aclContext_ = nullptr;
    aclrtRunMode runMode_ = ACL_DEVICE;
#endif
    std::map<std::string, std::map<std::string, ModulesInfo>> pipelineMap_ = {};
    ConfigParser configParser_ = {};
    int moduleTypeCount_ = 0;
    int moduleConnectCount_ = 0;
    ModuleConnectDesc *connnectDesc_ = nullptr;
};
}

#endif
