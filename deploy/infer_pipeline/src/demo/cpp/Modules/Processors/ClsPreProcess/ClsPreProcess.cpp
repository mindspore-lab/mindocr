/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Cls pre process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <cmath>
#include "ClsPreProcess.h"
#include "ClsInferProcess/ClsInferProcess.h"
#include "Utils.h"

using namespace ascendBaseModule;

ClsPreProcess::ClsPreProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

ClsPreProcess::~ClsPreProcess() {}

APP_ERROR ClsPreProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "ClsPreProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }
    clsHeight = 48;
    clsWidth = 192;

    LogInfo << "ClsPreProcess [" << instanceId_ << "] Init success.";
    return APP_ERR_OK;
}

APP_ERROR ClsPreProcess::DeInit(void)
{
    LogInfo << "ClsPreProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR ClsPreProcess::ParseConfig(ConfigParser &configParser)
{
    std::string clsModelPath;
    APP_ERROR ret = configParser.GetStringValue("clsModelPath", clsModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Parse the config file path failed, please check if the path is correct.";
        return ret;
    }
    std::string baseName = Utils::BaseName(clsModelPath) + ".bin";
    std::string modelConfigPath("./temp/cls/");
    Utils::LoadFromFileVec(modelConfigPath + baseName, batchSizeList);

    return APP_ERR_OK;
}

uint8_t *ClsPreProcess::PreprocessCls(std::vector<cv::Mat> &frames, uint32_t BatchSize)
{
    cv::Mat resizedImg;
    cv::Mat inImg;

    uint32_t bufferlen = Utils::RgbImageSizeF32(clsWidth, clsHeight);
    auto *srcData = new uint8_t[bufferlen * BatchSize];

    int pos = 0;
    for (uint32_t i = 0; i < frames.size(); i++) {
        inImg = frames[i];
        float ratio = float(inImg.cols) / float(inImg.rows);
        int resize_w;
        if (std::ceil(clsHeight * ratio) > clsWidth)
            resize_w = clsWidth;
        else
            resize_w = int(std::ceil(clsHeight * ratio));
        cv::resize(inImg, resizedImg, cv::Size(resize_w, clsHeight), 0.f, 0.f, cv::INTER_LINEAR);
        if (resize_w < clsWidth) {
            cv::copyMakeBorder(resizedImg, resizedImg, 0, 0, 0, clsWidth - resize_w, cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
        }

        resizedImg.convertTo(resizedImg, CV_32FC3, 1.0 / 255);
        resizedImg = (resizedImg - 0.5) / 0.5;

        std::vector<cv::Mat> channels;
        cv::split(resizedImg, channels);

        // Transform NHWC to NCHW
        uint32_t size = Utils::RgbImageSizeF32(clsWidth, clsHeight);
        uint8_t *buffer = Utils::ImageNchw(channels, size);

        // 把padding后的图片都组装起来
        memcpy(srcData + pos, buffer, bufferlen);
        pos += bufferlen;
        delete[] buffer;
    }

    return srcData;
}

std::vector<uint32_t> ClsPreProcess::GetClsBatchSize(uint32_t frameSize)
{
    int lastIndex = batchSizeList.size() - 1;
    std::vector<uint32_t> splitList(frameSize / batchSizeList[lastIndex], batchSizeList[lastIndex]);
    frameSize = frameSize - batchSizeList[lastIndex] * (frameSize / batchSizeList[lastIndex]);
    if (!frameSize) {
        return splitList;
    }
    for (auto bs : batchSizeList) {
        if (frameSize <= bs) {
            splitList.push_back(bs);
            break;
        }
    }
    return splitList;
}

APP_ERROR ClsPreProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    uint32_t totalSize = data->imgMatVec.size();
    if (totalSize == 0) {
        data->eof = true;
        SendToNextModule(MT_ClsInferProcess, data, data->channelId);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        Signal::clsPreProcessTime += costTime;
        Signal::e2eProcessTime += costTime;
        return APP_ERR_OK;
    }

    std::vector<uint32_t> splitIndex = GetClsBatchSize(totalSize);
    int startIndex = 0;
    for (unsigned int &bs : splitIndex) {
        std::shared_ptr<CommonData> dataNew = std::make_shared<CommonData>();

        std::vector<ResizedImageInfo> resizedImageInfosCls;
        std::vector<cv::Mat> input(data->imgMatVec.begin() + startIndex,
            data->imgMatVec.begin() + std::min(startIndex + bs, totalSize));
        std::vector<std::string> splitRes(data->inferRes.begin() + startIndex,
            data->inferRes.begin() + std::min(startIndex + bs, totalSize));

        uint8_t *ClsInput = PreprocessCls(input, bs);

        dataNew->eof = false;
        dataNew->outputTensorVec = data->outputTensorVec;
        dataNew->imgName = data->imgName;
        dataNew->inferRes = splitRes;
        dataNew->imgTotal = data->imgTotal;
        dataNew->imgMatVec = input;

        dataNew->resizedImageInfos = resizedImageInfosCls;
        dataNew->batchSize = bs;
        dataNew->imgBuffer = ClsInput;
        dataNew->frameSize = std::min(startIndex + bs, totalSize) - startIndex;
        dataNew->maxWHRatio = data->maxWHRatio;
        dataNew->saveFileName = data->saveFileName;
        dataNew->subImgTotal = data->subImgTotal;
        dataNew->imgId = data->imgId;

        startIndex += bs;
        SendToNextModule(MT_ClsInferProcess, dataNew, data->channelId);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Signal::clsPreProcessTime += costTime;
    Signal::e2eProcessTime += costTime;

    return APP_ERR_OK;
}
