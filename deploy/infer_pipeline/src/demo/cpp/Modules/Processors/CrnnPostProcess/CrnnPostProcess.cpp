/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Crnn post process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "CrnnPostProcess.h"
#include "CollectProcess/CollectProcess.h"
#include "MxBase/MxBase.h"
#include "Utils.h"

using namespace ascendBaseModule;

CrnnPostProcess::CrnnPostProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

CrnnPostProcess::~CrnnPostProcess() {}

APP_ERROR CrnnPostProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "CrnnPostProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }
    crnnPost_.ClassNameInit(recDictionary);

    LogInfo << "CrnnPostProcess [" << instanceId_ << "] Init success.";
    return APP_ERR_OK;
}

APP_ERROR CrnnPostProcess::DeInit(void)
{
    LogInfo << "CrnnPostProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR CrnnPostProcess::ParseConfig(ConfigParser &configParser)
{
    APP_ERROR ret = configParser.GetStringValue("dictPath", recDictionary);
    if (ret != APP_ERR_OK) {
        LogError << "Get dictPath failed, please check the value of dictPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(recDictionary, "character label file");
    if (ret != APP_ERR_OK) {
        LogError << "Character label file: " << recDictionary << " is not exist of can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    LogDebug << "dictPath: " << recDictionary;

    configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (saveInferResult) {
        configParser.GetStringValue("resultPath", resultPath);
        if (resultPath[resultPath.size() - 1] != '/') {
            resultPath += "/";
        }
    }

    return APP_ERR_OK;
}

APP_ERROR CrnnPostProcess::PostProcessCrnn(uint32_t framesSize, std::vector<MxBase::Tensor> &inferOutput,
    std::vector<std::string> &textsInfos)
{
    auto *objectinfo = (int64_t *)inferOutput[0].GetData();
    auto objectNum = (size_t)inferOutput[0].GetShape()[1];
    crnnPost_.CalcOutputIndex(objectinfo, framesSize, objectNum, textsInfos);

    return APP_ERR_OK;
}

APP_ERROR CrnnPostProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    std::vector<std::string> recResVec;
    if (!data->eof) {
        APP_ERROR ret = PostProcessCrnn(data->frameSize, data->outputTensorVec, data->inferRes);
        if (ret != APP_ERR_OK) {
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    Signal::recPostProcessTime += costTime;
    Signal::e2eProcessTime += costTime;
    SendToNextModule(MT_CollectProcess, data, data->channelId);
    return APP_ERR_OK;
}