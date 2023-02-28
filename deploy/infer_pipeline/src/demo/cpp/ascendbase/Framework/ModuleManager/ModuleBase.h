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

#ifndef INC_MODULE_BASE_H
#define INC_MODULE_BASE_H

#include <thread>
#include <vector>
#include <map>
#include <atomic>
#include "ConfigParser/ConfigParser.h"
#include "BlockingQueue/BlockingQueue.h"
#ifdef ASCEND_MODULE_USE_ACL
#include "acl/acl.h"
#endif

namespace ascendBaseModule {
enum ModuleConnectType {
    MODULE_CONNECT_ONE = 0,
    MODULE_CONNECT_CHANNEL, //
    MODULE_CONNECT_PAIR,    //
    MODULE_CONNECT_RANDOM   //
};

struct ModuleInitArguments {
#ifdef ASCEND_MODULE_USE_ACL
    aclrtRunMode runMode;
    aclrtContext context;
#endif
    std::string pipelineName = {};
    std::string moduleName = {};
    int instanceId = -1;
    void *userData = nullptr;
};

struct ModuleOutputInformation {
    std::string moduleName = "";
    ModuleConnectType connectType = MODULE_CONNECT_RANDOM;
    std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec = {};
    uint32_t outputQueVecSize = 0;
};

using ModuleInitArgs = ModuleInitArguments;
using ModuleOutputInfo = ModuleOutputInformation;

class ModuleBase {
public:
    ModuleBase() {};
    virtual ~ModuleBase() {};
    virtual APP_ERROR Init(ConfigParser &configParser, ModuleInitArgs &initArgs) = 0;
    virtual APP_ERROR DeInit(void) = 0;
    APP_ERROR Run(void); // create and run process thread
    APP_ERROR Stop(void);
    void SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue);
    void SetOutputInfo(std::string moduleName, ModuleConnectType connectType,
        std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);
    void SendToNextModule(std::string moduleNext, std::shared_ptr<void> outputData, int channelId = 0);
    const std::string GetModuleName();
    const int GetInstanceId();

public:
#ifdef ASCEND_MODULE_USE_ACL
    aclrtRunMode runMode_ = {};
    aclrtContext aclContext_ = {};
#endif

protected:
    void ProcessThread();
    virtual APP_ERROR Process(std::shared_ptr<void> inputData) = 0;
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
    int outputQueVecSize_ = 0;
    ModuleConnectType connectType_ = MODULE_CONNECT_RANDOM;
    int sendCount_ = 0;
};
}

#endif
