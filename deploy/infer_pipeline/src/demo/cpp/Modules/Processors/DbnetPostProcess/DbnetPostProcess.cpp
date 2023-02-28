/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Dbnet post process file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "DbnetPostProcess.h"
#include "CrnnPreProcess/CrnnPreProcess.h"
#include "ClsPreProcess//ClsPreProcess.h"
#include "DbnetPost.h"
#include "Utils.h"

using namespace ascendBaseModule;

DbnetPostProcess::DbnetPostProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

DbnetPostProcess::~DbnetPostProcess() {}

APP_ERROR DbnetPostProcess::Init(ConfigParser &configParser, ModuleInitArgs &initArgs)
{
    LogInfo << "Begin to init instance " << initArgs.instanceId;
    AssignInitArgs(initArgs);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "DbnetPostProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }

    bool useCls = Utils::EndsWith(initArgs.pipelineName, "true");
    if (useCls) {
        nextModule = MT_ClsPreProcess;
    } else {
        nextModule = MT_CrnnPreProcess;
    }
    LogInfo << "DbnetPostProcess [" << instanceId_ << "] Init success.";
    return APP_ERR_OK;
}

APP_ERROR DbnetPostProcess::DeInit(void)
{
    LogInfo << "DbnetPostProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR DbnetPostProcess::ParseConfig(ConfigParser &configParser)
{
    APP_ERROR ret = configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (ret != APP_ERR_OK) {
        LogError << "Get saveInferResult failed, please check the value of saveInferResult";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (saveInferResult) {
        ret = configParser.GetStringValue("resultPath", resultPath);
        if (ret != APP_ERR_OK) {
            LogError << "Get resultPath failed, please check the value of resultPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        if (access(resultPath.c_str(), 0) == -1) {
            int retCode = system(("mkdir -p " + resultPath).c_str());
            if (retCode == -1) {
                LogError << "Can not create dir [" << resultPath << "], please check the value of resultPath.";
                return APP_ERR_COMM_INVALID_PARAM;
            }
            LogInfo << resultPath << " create!";
        }
    }

    return APP_ERR_OK;
}


APP_ERROR DbnetPostProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);

    std::vector<ResizedImageInfo> resizedImageInfos;
    ResizedImageInfo ResizedInfo;

    ResizedInfo.widthResize = data->resizeWidth;
    ResizedInfo.heightResize = data->resizeHeight;
    ResizedInfo.widthOriginal = data->srcWidth;
    ResizedInfo.heightOriginal = data->srcHeight;
    ResizedInfo.ratio = data->ratio;
    resizedImageInfos.emplace_back(std::move(ResizedInfo));

    std::vector<std::vector<TextObjectInfo>> textObjInfos;

    DbnetPost DbnetPost;
    DbnetPost.DbnetObjectDetectionOutput(data->outputTensorVec, textObjInfos, resizedImageInfos);

    std::vector<cv::Mat> resizeImgs;
    std::vector<std::string> inferRes;
    float maxWHRatio = 0;

    for (uint32_t i = 0; i < textObjInfos.size(); i++) {
        std::vector<TextObjectInfo> textInfo = textObjInfos[i];
        for (uint32_t j = 0; j < textInfo.size(); j++) {
            LogDebug << "#Obj " << j;
            LogDebug << "x0 " << textInfo[j].x0 << " y0 " << textInfo[j].y0;
            LogDebug << "x1 " << textInfo[j].x1 << " y1 " << textInfo[j].y1;
            LogDebug << "x2 " << textInfo[j].x2 << " y2 " << textInfo[j].y2;
            LogDebug << "x3 " << textInfo[j].x3 << " y3 " << textInfo[j].y3;
            LogDebug << "confidence: " << textInfo[j].confidence;

            cv::Mat resizeimg;
            std::string str = std::to_string((int)textInfo[j].x1) + "," + std::to_string((int)textInfo[j].y1) + "," +
                std::to_string((int)textInfo[j].x2) + "," + std::to_string((int)textInfo[j].y2) + "," +
                std::to_string((int)textInfo[j].x3) + "," + std::to_string((int)textInfo[j].y3) + "," +
                std::to_string((int)textInfo[j].x0) + "," + std::to_string((int)textInfo[j].y0) + ",";
            inferRes.push_back(str);

            float cropWidth = CalcCropWidth(textInfo[j]);
            float cropHeight = CalcCropHeight(textInfo[j]);

            // 期望透视变换后二维码四个角点的坐标
            cv::Point2f dst_points[4];
            cv::Point2f src_points[4];
            // 通过Image Watch查看的二维码四个角点坐标
            src_points[0] = cv::Point2f(textInfo[j].x0, textInfo[j].y0);
            src_points[1] = cv::Point2f(textInfo[j].x1, textInfo[j].y1);
            src_points[2] = cv::Point2f(textInfo[j].x2, textInfo[j].y2);
            src_points[3] = cv::Point2f(textInfo[j].x3, textInfo[j].y3);

            dst_points[0] = cv::Point2f(0.0, 0.0);
            dst_points[1] = cv::Point2f(cropWidth, 0.0);
            dst_points[2] = cv::Point2f(cropWidth, cropHeight);
            dst_points[3] = cv::Point2f(0.0, cropHeight);

            cv::Mat H = cv::getPerspectiveTransform(src_points, dst_points);

            cv::Mat rotation;

            cv::Mat img_warp = cv::Mat(cropHeight, cropWidth, CV_8UC3);
            cv::warpPerspective(data->frame, img_warp, H, img_warp.size());
            int imgH = img_warp.rows;
            int imgW = img_warp.cols;
            if (imgH * 1.0 / imgW >= 1.5) {
                cv::rotate(img_warp, img_warp, cv::ROTATE_90_COUNTERCLOCKWISE);
                imgH = img_warp.rows;
                imgW = img_warp.cols;
            }
            maxWHRatio = std::max(maxWHRatio, float(imgW) / float(imgH));
            resizeImgs.emplace_back(img_warp);
        }
    }
    data->maxWHRatio = maxWHRatio;
    data->imgMatVec = resizeImgs;
    data->inferRes = inferRes;
    data->subImgTotal = resizeImgs.size();
    SendToNextModule(nextModule, data, data->channelId);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Signal::detPostProcessTime += costTime;
    Signal::e2eProcessTime += costTime;

    return APP_ERR_OK;
}

float DbnetPostProcess::CalcCropWidth(const TextObjectInfo &textObject)
{
    float x0 = std::abs(textObject.x1 - textObject.x0);
    float y0 = std::abs(textObject.y1 - textObject.y0);
    float line0 = sqrt(std::pow(x0, 2) + std::pow(y0, 2));

    float x1 = std::abs(textObject.x2 - textObject.x3);
    float y1 = std::abs(textObject.y2 - textObject.y3);
    float line1 = std::sqrt(std::pow(x1, 2) + std::pow(y1, 2));

    return line1 > line0 ? line1 : line0;
}

float DbnetPostProcess::CalcCropHeight(const TextObjectInfo &textObject)
{
    float x0 = std::abs(textObject.x0 - textObject.x3);
    float y0 = std::abs(textObject.y0 - textObject.y3);
    float line0 = sqrt(std::pow(x0, 2) + std::pow(y0, 2));

    float x1 = std::abs(textObject.x1 - textObject.x2);
    float y1 = std::abs(textObject.y1 - textObject.y2);
    float line1 = std::sqrt(std::pow(x1, 2) + std::pow(y1, 2));

    return line1 > line0 ? line1 : line0;
}
