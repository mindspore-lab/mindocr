/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Dbnet infer process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_DBNETINFERPROCESS_H
#define CPP_DBNETINFERPROCESS_H

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "Signal.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class DbnetInferProcess : public ascendBaseModule::ModuleBase {
public:
    DbnetInferProcess();

    ~DbnetInferProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int32_t deviceId_ = 0;
    std::unique_ptr<MxBase::Model> dbNet_;
    std::vector<MxBase::Tensor> dbNetoutputs;

    APP_ERROR ParseConfig(ConfigParser &configParser);
};

MODULE_REGIST(DbnetInferProcess)

#endif
