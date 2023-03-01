/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: collect profiling and infer result data process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_COLLECTPROCESS_H
#define CPP_COLLECTPROCESS_H

#include <unordered_map>

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Signal.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"


class CollectProcess : public ascendBaseModule::ModuleBase {
public:
    CollectProcess();

    ~CollectProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    std::string resultPath;
    bool saveInferResult = false;

    std::unordered_map<int, int> idMap;
    int inferSize = 0;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    void SignalSend(int imgTotal);
};

MODULE_REGIST(CollectProcess)

#endif
