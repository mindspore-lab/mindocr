/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Cls infer process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_CLSINFERPROCESS_H
#define CPP_CLSINFERPROCESS_H

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "Signal.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class ClsInferProcess : public ascendBaseModule::ModuleBase {
public:
    ClsInferProcess();

    ~ClsInferProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int clsHeight;
    int clsWidth;
    int32_t deviceId_ = 0;
    std::unique_ptr<MxBase::Model> ClsNet_;
    std::vector<uint32_t> batchSizeList;
    std::vector<MxBase::Tensor> ClsOutputs;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    std::vector<MxBase::Tensor> ClsModelInfer(uint8_t *srcData, uint32_t BatchSize, int maxResizedW);
};

MODULE_REGIST(ClsInferProcess)

#endif
