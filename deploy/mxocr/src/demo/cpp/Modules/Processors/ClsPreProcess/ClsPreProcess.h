/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Cls pre process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_CLSPREPROCESS_H
#define CPP_CLSPREPROCESS_H

#include "ModuleManager/ModuleManager.h"
#include "ConfigParser/ConfigParser.h"
#include "Utils.h"
#include "DataType/DataType.h"
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"
#include "Signal.h"

class ClsPreProcess : public ascendBaseModule::ModuleBase {
public:
    ClsPreProcess();

    ~ClsPreProcess();

    APP_ERROR Init(ConfigParser &configParser, ascendBaseModule::ModuleInitArgs &initArgs);

    APP_ERROR DeInit(void);

protected:
    APP_ERROR Process(std::shared_ptr<void> inputData);

private:
    int clsHeight = 48;
    int clsWidth = 192;
    int32_t deviceId_ = 0;
    std::vector<uint64_t> batchSizeList;

    APP_ERROR ParseConfig(ConfigParser &configParser);

    std::vector<uint32_t> GetClsBatchSize(uint32_t frameSize);

    uint8_t *PreprocessCls(std::vector<cv::Mat> &frames, uint32_t batchSize);
};

MODULE_REGIST(ClsPreProcess)

#endif
