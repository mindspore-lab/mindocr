/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: collect profiling and infer result data process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "CollectProcess.h"
#include "Utils.h"

using namespace ascendBaseModule;

CollectProcess::CollectProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

CollectProcess::~CollectProcess() {}

APP_ERROR CollectProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "CollectProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    LogInfo << "CollectProcess [" << instanceId_ << "] Init success.";
    return APP_ERR_OK;
}

APP_ERROR CollectProcess::DeInit(void)
{
    LogInfo << "CollectProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR CollectProcess::ParseConfig(ConfigParser &configParser)
{
    configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (saveInferResult) {
        configParser.GetStringValue("resultPath", resultPath);
        if (resultPath[resultPath.size() - 1] != '/') {
            resultPath += "/";
        }
    }

    return APP_ERR_OK;
}

void CollectProcess::SignalSend(int imgTotal)
{
    if (inferSize == imgTotal) {
        Signal::GetInstance().GetStopedThreadNum()++;
        if (Signal::GetInstance().GetStopedThreadNum() == Signal::GetInstance().GetThreadNum()) {
            Signal::signalRecieved = true;
        }
    }
}

APP_ERROR CollectProcess::Process(std::shared_ptr<void> commonData)
{
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    if (saveInferResult) {
        std::ofstream outfile(resultPath + data->saveFileName, std::ios::out | std::ios::app);
        std::string line;
        for (uint32_t i = 0; i < data->frameSize; i++) {
            std::string finalRes = data->inferRes[i];
            outfile << finalRes << std::endl;
        }
        outfile.close();
        LogInfo << "----------------------- Save infer result to " << data->saveFileName << " succeed.";
    }

    auto it = idMap.find(data->imgId);
    if (it == idMap.end()) {
        int remaining = data->subImgTotal - data->inferRes.size();
        if (remaining) {
            idMap.insert({ data->imgId, remaining });
        } else {
            inferSize += 1;
        }
    } else {
        it->second -= data->inferRes.size();
        if (it->second == 0) {
            idMap.erase(it);
            inferSize += 1;
        }
    }


    SignalSend(data->imgTotal);
    return APP_ERR_OK;
}