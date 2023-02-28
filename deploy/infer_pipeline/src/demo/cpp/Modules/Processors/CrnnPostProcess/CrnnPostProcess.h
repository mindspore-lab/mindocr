/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Crnn post process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_CRNNPOSTPROCESS_H
#define CPP_CRNNPOSTPROCESS_H

#include <unordered_set>

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Signal.h"
#include "DataType/DataType.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

#include "CrnnPost.h"

class CrnnPostProcess : public ascendBaseModule::ModuleBase {
public:
    CrnnPostProcess();

    ~CrnnPostProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    CrnnPost crnnPost_;
    std::string recDictionary;
    std::string resultPath;
    bool saveInferResult = false;

    std::unordered_set<int> idSet;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    APP_ERROR PostProcessCrnn(uint32_t framesSize, std::vector<MxBase::Tensor> &inferOutput,
        std::vector<std::string> &textsInfos);
};

MODULE_REGIST(CrnnPostProcess)

#endif
