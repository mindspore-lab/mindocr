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

#include "CrnnPost.h"
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stack>
#include "Log/Log.h"

CrnnPost::CrnnPost(void) {}

// trim from end of string (right)
inline std::string &rtrim(std::string &s)
{
    const char *ws = "\n\r";
    s.erase(s.find_last_not_of(ws) + 1);
    return s;
}

void CrnnPost::ClassNameInit(const std::string &fileName)
{
    std::ifstream fsIn(fileName);
    std::string line;

    if (fsIn) { // 有该文件
        labelVec_.clear();
        labelVec_.push_back("");
        while (getline(fsIn, line)) {
            labelVec_.push_back(rtrim(line));
        }
    } else {
        LogInfo << "no such file";
    }
    labelVec_.push_back(" ");
    fsIn.close();

    classNum_ = labelVec_.size();

    LogInfo << " ClassNameInit classNum_ " << classNum_;

    return;
}

std::string CrnnPost::GetClassName(const size_t classId)
{
    if (classId >= labelVec_.size()) {
        LogError << "Failed to get classId(" << classId << ") label, size(" << labelVec_.size() << ").";
        return "";
    }
    return labelVec_[classId];
}


void CrnnPost::CalcOutputIndex(void *resHostBuf, size_t batchSize, size_t objectNum, std::vector<std::string> &resVec)
{
    LogDebug << "Start to Process CalcOutputIndex.";

    auto *outputInfo = static_cast<int64_t *>(resHostBuf);

    batchSize_ = batchSize;
    objectNum_ = objectNum;

    for (uint32_t i = 0; i < batchSize_; i++) {
        int64_t blankIdx = blankIdx_;
        int64_t previousIdx = blankIdx_;
        std::string result = "";
        std::string str = "";
        for (int64_t j = 0; j < objectNum_; j++) {
            int64_t outputNum = i * objectNum_ + j;
            if (outputInfo[outputNum] != blankIdx && outputInfo[outputNum] != previousIdx) {
                result = GetClassName(outputInfo[outputNum]);
                str += result;
            }
            previousIdx = outputInfo[outputNum];
        }
        if (std::strcmp(str.c_str(), "") == 0) {
            str = "###";
        }

        resVec[i] = resVec[i] + str;
    }
}
