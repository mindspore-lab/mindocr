/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Crnn infer process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_CRNNINFERPROCESS_H
#define CPP_CRNNINFERPROCESS_H

#include "CrnnPost.h"

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "DataType/DataType.h"
#include "Signal.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class CrnnInferProcess : public ascendBaseModule::ModuleBase {
public:
    CrnnInferProcess();

    ~CrnnInferProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int mStdHeight = 48;
    int32_t deviceId_ = 0;
    bool staticMethod = true;
    std::vector<MxBase::Model *> crnnNet_;
    std::vector<uint32_t> batchSizeList;
    std::vector<MxBase::Tensor> crnnOutputs;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    std::vector<MxBase::Tensor> CrnnModelInfer(uint8_t *srcData, uint32_t BatchSize, int maxResizedW);
};

MODULE_REGIST(CrnnInferProcess)

#endif
