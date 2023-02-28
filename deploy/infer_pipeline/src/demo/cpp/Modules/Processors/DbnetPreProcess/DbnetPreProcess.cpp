/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Dbnet pre process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "DbnetPreProcess.h"
#include "DbnetInferProcess/DbnetInferProcess.h"

#include "MxBase/MxBase.h"
#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>

using namespace ascendBaseModule;

DbnetPreProcess::DbnetPreProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

DbnetPreProcess::~DbnetPreProcess() {}

APP_ERROR DbnetPreProcess::Init(ConfigParser &confiPgarser, ModuleInitArgs &initArgs)
{
    LogInfo << "Begin to init instance " << initArgs.instanceId;
    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(confiPgarser);
    if (ret != APP_ERR_OK) {
        LogError << "DbnetPreProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    if (deviceType_ == "310P") {
        imageProcessor.reset(new MxBase::ImageProcessor(deviceId_));
    }
    LogInfo << "DbnetPreProcess [" << instanceId_ << "] Init success.";
    return APP_ERR_OK;
}

APP_ERROR DbnetPreProcess::DeInit(void)
{
    LogInfo << "DbnetPreProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR DbnetPreProcess::ParseConfig(ConfigParser &configParser)
{
    APP_ERROR ret = configParser.GetStringValue("deviceType", deviceType_);
    if (ret != APP_ERR_OK) {
        LogError << "Get device type failed, please check the value of deviceType";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (deviceType_ != "310P" && deviceType_ != "310") {
        LogError << "Device type only support 310 or 310P, please check the value of device type.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::vector<uint32_t> deviceIdVec;
    ret = configParser.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get device id failed, please check the value of deviceId";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    deviceId_ = (int32_t)deviceIdVec[instanceId_ % deviceIdVec.size()];
    if (deviceId_ < 0) {
        LogError << "Device id: " << deviceId_ << " is less than 0, not valid";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::string detModelPath;
    ret = configParser.GetStringValue("detModelPath", detModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Parse the config file path failed, please check if the path is correct.";
        return ret;
    }

    std::string baseName = Utils::BaseName(detModelPath) + ".1.bin";
    std::string modelConfigPath("./temp/dbnet/");
    Utils::LoadFromFilePair(modelConfigPath + baseName, gearInfo);
    std::sort(gearInfo.begin(), gearInfo.end(), Utils::PairCompare);
    uint64_t hwSum = 0;
    for (auto &pair : gearInfo) {
        uint64_t h = pair.first;
        uint64_t w = pair.second;
        MaxH = MaxH > h ? MaxH : h;
        MaxW = MaxW > w ? MaxW : w;
        if (h * w > hwSum) {
            hwSum = h * w;
            maxDotGear.first = h;
            maxDotGear.second = w;
        }
    }

    return APP_ERR_OK;
}

void DbnetPreProcess::getMatchedGear(const cv::Mat &inImg, std::pair<uint64_t, uint64_t> &gear)
{
    uint64_t imgH = inImg.rows;
    uint64_t imgW = inImg.cols;
    if (imgH > MaxH || imgW > MaxW) {
        gear = maxDotGear;
    } else {
        auto info = std::upper_bound(gearInfo.begin(), gearInfo.end(), std::pair<uint64_t, uint64_t>(imgH, imgW),
            Utils::GearCompare);
        gear = gearInfo[info - gearInfo.begin()];
    }
}

void DbnetPreProcess::resize(const cv::Mat &inImg, cv::Mat &outImg, const std::pair<uint64_t, uint64_t> &gear,
    float &ratio_)
{
    int imgH = inImg.rows;
    int imgW = inImg.cols;
    int gearH = gear.first;
    int gearW = gear.second;
    float ratio = 1.f;
    if (imgH > gearH || imgW > gearW) {
        if (imgH > imgW) {
            ratio = float(gearH) / float(imgH);
            int resizeByH = int(ratio * float(imgW));
            if (resizeByH > gearW) {
                ratio = float(gearW) / float(imgW);
            }
        } else {
            ratio = float(gearW) / float(imgW);
            int resizeByW = int(ratio * float(imgH));
            if (resizeByW > gearH) {
                ratio = float(gearH) / float(imgH);
            }
        }
    }
    int resizeH = int(float(imgH) * ratio);
    int resizeW = int(float(imgW) * ratio);
    cv::resize(inImg, outImg, cv::Size(resizeW, resizeH));
    ratio_ = float(resizeH) / float(imgH);
}

void DbnetPreProcess::padding(cv::Mat &inImg, const std::pair<uint64_t, uint64_t> &gear)
{
    int imgH = inImg.rows;
    int imgW = inImg.cols;
    int gearH = gear.first;
    int gearW = gear.second;
    int paddingH = gearH - imgH;
    int paddingW = gearW - imgW;
    if (paddingH || paddingW) {
        cv::copyMakeBorder(inImg, inImg, 0, paddingH, 0, paddingW, cv::BORDER_CONSTANT, 0);
    }
}

cv::Mat DbnetPreProcess::DecodeImgDvpp(std::string imgPath)
{
    MxBase::Image decodedImage;
    imageProcessor->Decode(imgPath, decodedImage, MxBase::ImageFormat::BGR_888);
    decodedImage.ToHost();

    MxBase::Size imgOriSize = decodedImage.GetOriginalSize();
    MxBase::Size imgSize = decodedImage.GetSize();
    cv::Mat imgBGR;
    imgBGR.create(imgSize.height, imgSize.width, CV_8UC3);
    imgBGR.data = (uchar *)decodedImage.GetData().get();
    cv::Rect area(0, 0, imgOriSize.width, imgOriSize.height);
    imgBGR = imgBGR(area).clone();
    return imgBGR;
}

void DbnetPreProcess::normalizeByChannel(std::vector<cv::Mat> &bgr_channels)
{
    for (uint32_t i = 0; i < bgr_channels.size(); i++) {
        bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale_[i], (0.0 - mean_[i]) * scale_[i]);
    }
}

APP_ERROR DbnetPreProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    std::string imgPath = data->imgPath;

    std::chrono::high_resolution_clock::time_point dbPreEndTime;
    cv::Mat inImg;
    if (deviceType_ == "310P") {
        inImg = DecodeImgDvpp(imgPath);
    } else {
        inImg = cv::imread(imgPath);
    }
    data->frame = inImg;
    data->srcWidth = inImg.cols;
    data->srcHeight = inImg.rows;
    cv::Mat resizedImg;
    cv::Mat outImg;

    inImg.convertTo(inImg, CV_32FC3, 1.0 / 255.0);
    std::pair<uint64_t, uint64_t> gear;
    getMatchedGear(inImg, gear);

    float ratio = 0;
    resize(inImg, resizedImg, gear, ratio);

    padding(resizedImg, gear);

    // Normalize: y = (x - mean) / std
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(resizedImg, bgr_channels);
    normalizeByChannel(bgr_channels);

    // Transform NHWC to NCHW
    uint32_t size = Utils::RgbImageSizeF32(resizedImg.cols, resizedImg.rows);
    uint8_t *buffer = Utils::ImageNchw(bgr_channels, size);

    data->eof = false;
    data->channelId = 0;
    data->imgBuffer = buffer;
    data->resizeWidth = resizedImg.cols;
    data->resizeHeight = resizedImg.rows;
    data->ratio = ratio;
    SendToNextModule(MT_DbnetInferProcess, data, data->channelId);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Signal::detPreProcessTime += costTime;
    Signal::e2eProcessTime += costTime;

    return APP_ERR_OK;
}
