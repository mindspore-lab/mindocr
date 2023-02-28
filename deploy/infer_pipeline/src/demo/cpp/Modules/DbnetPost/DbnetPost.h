/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DBNETPOST_H
#define DBNETPOST_H

#include <opencv2/imgproc.hpp>
#include "Log/Log.h"
#include "Utils/Utils.h"
#include "ErrorCode/ErrorCode.h"

const float THRESH = 0.3;
const float BOXTHRESH = 0.5;
const int MIN_SIZE = 3;
const int MAX_SIZE = 5;
const int POINT1 = 0;
const int POINT2 = 1;
const int POINT3 = 2;
const int POINT4 = 3;
const int POINTNUM = 4;
const int INDEX2 = 2;
const int MAX_CANDIDATES = 999;
const int MAX_VAL = 255;
const float UNCLIP_RATIO = 2;
const int UNCLIP_DISTANCE = 2;

class DbnetPost {
public:
    DbnetPost(void);

    ~DbnetPost() {};

    APP_ERROR DbnetObjectDetectionOutput(std::vector<MxBase::Tensor> &singleResult,
        std::vector<std::vector<TextObjectInfo>> &textObjInfos, const std::vector<ResizedImageInfo> &resizedImageInfos);

private:
    void FilterByMinSize(std::vector<cv::Point> &contour, std::vector<cv::Point2f> &box, float &minSide);

    void FilterByBoxScore(const cv::Mat &prediction, std::vector<cv::Point2f> &box, float &score);

    void FilterByClippedMinSize(std::vector<cv::Point2f> &box, float &minSide);

    void ConstructInfo(std::vector<TextObjectInfo> &textObjectInfo, std::vector<cv::Point2f> &box,
        const std::vector<ResizedImageInfo> &resizedImageInfos, const uint32_t &index, float score);

    const int NpClip(const int &coordinate, const int &sideLen);

    float PointsL2Distance(cv::Point2f p1, cv::Point2f p2);

    static bool SortByX(cv::Point2f p1, cv::Point2f p2);

    static bool SortByY(cv::Point2f p1, cv::Point2f p2);

    int minSize_ = MIN_SIZE;
    float thresh_ = THRESH;
    float boxThresh_ = BOXTHRESH;
    uint32_t resizedW_;
    uint32_t resizedH_;

    float unclipRatio_ = UNCLIP_RATIO;
    int candidates_ = MAX_CANDIDATES;
};

#endif // DBNETPOST_H
