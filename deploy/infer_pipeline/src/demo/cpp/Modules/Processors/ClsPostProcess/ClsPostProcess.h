/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Cls post process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_CLSPOSTPROCESS_H
#define CPP_CLSPOSTPROCESS_H

#include <unordered_set>

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Signal.h"
#include "DataType/DataType.h"
#include "MxBase/MxBase.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class ClsPostProcess : public ascendBaseModule::ModuleBase {
public:
    ClsPostProcess();

    ~ClsPostProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    APP_ERROR PostProcessCls(uint32_t framesSize, std::vector<MxBase::Tensor> &inferOutput,
        std::vector<cv::Mat> &imgMatVec);
};

MODULE_REGIST(ClsPostProcess)

#endif
