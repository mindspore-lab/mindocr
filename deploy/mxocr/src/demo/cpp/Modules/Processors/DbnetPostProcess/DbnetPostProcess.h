/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Dbnet post process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_DBNETPOSTPROCESS_H
#define CPP_DBNETPOSTPROCESS_H

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"
#include "Signal.h"
#include "Log/Log.h"

class DbnetPostProcess : public ascendBaseModule::ModuleBase {
public:
    DbnetPostProcess();

    ~DbnetPostProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    bool saveInferResult = false;
    std::string resultPath;
    std::string nextModule;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    static float CalcCropWidth(const TextObjectInfo &textObject);

    static float CalcCropHeight(const TextObjectInfo &textObject);
};

MODULE_REGIST(DbnetPostProcess)

#endif
