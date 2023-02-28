/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Dbnet infer process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "DbnetInferProcess.h"
#include "DbnetPostProcess/DbnetPostProcess.h"
#include <iostream>

using namespace ascendBaseModule;

DbnetInferProcess::DbnetInferProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

DbnetInferProcess::~DbnetInferProcess() {}

APP_ERROR DbnetInferProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "Begin to init instance " << initArgs.instanceId;
    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "DbnetInferProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    LogInfo << "DbnetInferProcess [" << instanceId_ << "] Init success.";
    return ret;
}

APP_ERROR DbnetInferProcess::DeInit(void)
{
    LogInfo << "DbnetInferProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR DbnetInferProcess::ParseConfig(ConfigParser &configParser)
{
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = configParser.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get device id failed, please check the value of deviceId";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    deviceId_ = (int32_t)deviceIdVec[instanceId_ % deviceIdVec.size()];
    std::string detModelPath;
    ret = configParser.GetStringValue("detModelPath", detModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get detModelPath failed, please check the value of detModelPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(detModelPath, "detModelPath");
    if (ret != APP_ERR_OK) {
        LogError << "detModelPath: " << detModelPath << " is not exist or can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    LogDebug << "detModelPath: " << detModelPath;

    dbNet_.reset(new MxBase::Model(detModelPath, deviceId_));
    LogInfo << deviceId_;
    return APP_ERR_OK;
}

APP_ERROR DbnetInferProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);

    std::vector<uint32_t> shape;
    uint32_t batchSize = 1;
    shape.push_back(batchSize);
    shape.push_back(3);
    shape.push_back(data->resizeHeight);
    shape.push_back(data->resizeWidth);
    MxBase::TensorDType tensorDataType = MxBase::TensorDType::FLOAT32;

    MxBase::Tensor imageToTensor(data->imgBuffer, shape, tensorDataType, deviceId_);

    std::vector<MxBase::Tensor> inputs = {};
    inputs.push_back(imageToTensor);

    // (2) 开始推理
    auto inferStartTime = std::chrono::high_resolution_clock::now();
    dbNetoutputs = dbNet_->Infer(inputs);
    auto inferEndTime = std::chrono::high_resolution_clock::now();
    double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
    Signal::detInferTime += inferCostTime;
    LogInfo << " [" << data->imgName << "] dbnet infer time cost: " << inferCostTime << "ms.";

    LogDebug << "c: " << dbNetoutputs[0].GetShape()[1];
    LogDebug << "h: " << dbNetoutputs[0].GetShape()[2];
    LogDebug << "w: " << dbNetoutputs[0].GetShape()[3];
    LogDebug << "End Model Infer progress...";
    LogDebug << "current imgName: " << data->imgName;
    LogDebug << "current resizeHeight: " << data->resizeHeight;
    LogDebug << "current resizeWidth: " << data->resizeWidth;

    const size_t outputLen = dbNetoutputs.size();
    if (outputLen <= 0) {
        LogError << "Failed to get model output data";
        return APP_ERR_INFER_GET_OUTPUT_FAIL;
    }
    for (auto &output : dbNetoutputs) {
        output.ToHost();
    }


    std::vector<MxBase::Tensor> res = { dbNetoutputs[0] };
    data->outputTensorVec = res;


    if (data->imgBuffer != nullptr) {
        delete data->imgBuffer;
        data->imgBuffer = nullptr;
    }
    SendToNextModule(MT_DbnetPostProcess, data, data->channelId);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Signal::detInferProcessTime += costTime;
    Signal::e2eProcessTime += costTime;
    return APP_ERR_OK;
}
