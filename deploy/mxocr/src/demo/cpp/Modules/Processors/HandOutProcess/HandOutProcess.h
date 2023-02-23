/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Crnn infer process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_HANDOUTPROCESS_H
#define CPP_HANDOUTPROCESS_H

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "Utils.h"
#include "Signal.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class HandOutProcess : public ascendBaseModule::ModuleBase {
public:
    HandOutProcess();

    ~HandOutProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int imgId_ = 0;
    std::string deviceType_;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    bool saveInferResult;
    std::string resultPath;
};

MODULE_REGIST(HandOutProcess)

#endif
