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

#ifndef CRNNPOST_H
#define CRNNPOST_H

#include <algorithm>
#include <vector>
#include <memory>
#include "Utils/Utils.h"

class CrnnPost {
public:
    CrnnPost(void);

    ~CrnnPost() {};

    void ClassNameInit(const std::string &fileName);

    std::string GetClassName(const size_t classId);

    void CalcOutputIndex(void *resHostBuf, size_t batchSize, size_t objectNum, std::vector<std::string> &resVec);

private:
    std::vector<std::string> labelVec_ = {}; // labels info
    uint32_t classNum_ = 0;
    uint32_t objectNum_ = 200;
    uint32_t batchSize_ = 0;
    uint32_t blankIdx_ = 0;
};

#endif
