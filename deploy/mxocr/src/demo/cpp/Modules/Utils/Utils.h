/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Util data struct.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>

#include "MxBase/MxBase.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// Class for text object detection
class __attribute__((visibility("default"))) TextObjectInfo {
public:
    float x0;
    float y0;
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float confidence;
    std::string result;
};

// Class for resized image info
class __attribute__((visibility("default"))) ResizedImageInfo {
public:
    uint32_t widthResize;    // memoryWidth
    uint32_t heightResize;   // memoryHeight
    uint32_t widthOriginal;  // imageWidth
    uint32_t heightOriginal; // imageHeight
    float ratio;
};

// Class for text generation (i.e. translation, OCR)
class __attribute__((visibility("default"))) TextsInfo {
public:
    std::vector<std::string> text;
};

class Utils {
public:
    static uint32_t ImageChanSizeF32(uint32_t width, uint32_t height);

    static uint32_t RgbImageSizeF32(uint32_t width, uint32_t height);

    static uint8_t *ImageNchw(std::vector<cv::Mat> &nhwcImageChs, uint32_t size);

    static APP_ERROR CheckPath(const std::string &srcPath, const std::string &config);

    static std::string BaseName(const std::string &filename);

    static void LoadFromFilePair(const std::string &filename, std::vector<std::pair<uint64_t, uint64_t>> &vec);

    static void SaveToFilePair(const std::string &filename, std::vector<std::pair<uint64_t, uint64_t>> &vec);

    static void LoadFromFileVec(const std::string &filename, std::vector<uint64_t> &vec);

    static void SaveToFileVec(const std::string &filename, std::vector<uint64_t> &vec);

    static bool PairCompare(std::pair<uint64_t, uint64_t> p1, std::pair<uint64_t, uint64_t> p2);

    static bool GearCompare(std::pair<uint64_t, uint64_t> p1, std::pair<uint64_t, uint64_t> p2);

    static bool UintCompare(uint64_t num1, uint64_t num2);

    static bool ModelCompare(MxBase::Model *model1, MxBase::Model *model2);

    static void GetAllFiles(const std::string &dirName, std::vector<std::string> &files);
    static bool EndsWith(std::string const & value, std::string const & ending);

    static void StrSplit(const std::string &str, const std::string &pattern, std::vector<std::string> &vec);

    static void MakeDir(const std::string &path, bool replace);

    static std::string GenerateResName(const std::string &basename);

    static std::string BoolCast(bool b);
};

#endif
