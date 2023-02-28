/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Common data for modules.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_DATATYPE_H
#define CPP_DATATYPE_H

#include "Utils.h"
#include "MxBase/MxBase.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

struct CommonData {
    // modules info
    int channelId = {};
    int imgId = {};
    int imgTotal = {};
    int subImgTotal = {};

    // img info
    std::string imgPath = {};
    std::string imgName = {};
    uint32_t srcWidth = {};
    uint32_t srcHeight = {};
    cv::Mat frame = {};
    std::vector<cv::Mat> imgMatVec = {};

    // infer related
    uint8_t *imgBuffer = {};
    std::vector<MxBase::Tensor> outputTensorVec = {};
    std::vector<ResizedImageInfo> resizedImageInfos = {};

    // det info
    uint32_t resizeWidth = {};
    uint32_t resizeHeight = {};
    float ratio = {};

    // cls and rec
    uint32_t batchSize = {};
    uint32_t frameSize = {};
    bool eof = {};

    // rec
    int maxResizedW = {};
    float maxWHRatio = {};

    // infer res
    std::string saveFileName = {};
    std::vector<std::string> inferRes = {};
};

#endif
