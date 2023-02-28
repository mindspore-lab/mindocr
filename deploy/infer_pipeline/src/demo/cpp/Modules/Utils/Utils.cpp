/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Util data struct.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "Utils.h"

uint32_t Utils::ImageChanSizeF32(uint32_t width, uint32_t height)
{
    return width * height * 4;
}

uint32_t Utils::RgbImageSizeF32(uint32_t width, uint32_t height)
{
    return width * height * 3 * 4;
}

uint8_t *Utils::ImageNchw(std::vector<cv::Mat> &nhwcImageChs, uint32_t size)
{
    uint8_t *nchwBuf = new uint8_t[size];
    uint32_t channelSize = ImageChanSizeF32(nhwcImageChs[0].rows, nhwcImageChs[0].cols);
    int pos = 0;
    for (unsigned int i = 0; i < nhwcImageChs.size(); i++) {
        memcpy(static_cast<uint8_t *>(nchwBuf) + pos, nhwcImageChs[i].ptr<float>(0), channelSize);
        pos += channelSize;
    }

    return nchwBuf;
}

APP_ERROR Utils::CheckPath(const std::string &srcPath, const std::string &config)
{
    int folderExist = access(srcPath.c_str(), R_OK);
    if (folderExist == -1) {
        LogError << config << " doesn't exist or can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    return APP_ERR_OK;
}

std::string Utils::BaseName(const std::string &filename)
{
    if (filename.empty()) {
        return "";
    }

    auto len = filename.length();
    auto index = filename.find_last_of("/\\");

    if (index == std::string::npos) {
        return filename;
    }

    if (index + 1 >= len) {
        len--;
        index = filename.substr(0, len).find_last_of("/\\");

        if (len == 0) {
            return filename;
        }

        if (index == 0) {
            return filename.substr(1, len - 1);
        }

        if (index == std::string::npos) {
            return filename.substr(0, len);
        }

        return filename.substr(index + 1, len - index - 1);
    }

    return filename.substr(index + 1, len - index);
}

void Utils::LoadFromFilePair(const std::string &filename, std::vector<std::pair<uint64_t, uint64_t>> &vec)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    std::pair<uint64_t, uint64_t> pair;
    while (file.read(reinterpret_cast<char *>(&pair), sizeof(pair))) {
        vec.push_back(pair);
    }
}

void Utils::SaveToFilePair(const std::string &filename, std::vector<std::pair<uint64_t, uint64_t>> &vec)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    for (auto &inf : vec) {
        file.write(reinterpret_cast<const char *>(&inf), sizeof(inf));
    }
}

bool Utils::PairCompare(const std::pair<uint64_t, uint64_t> p1, const std::pair<uint64_t, uint64_t> p2)
{
    if (p1.first < p2.first) {
        return true;
    } else if (p1.first == p2.first) {
        return p1.second < p2.second;
    }
    return false;
}

bool Utils::GearCompare(const std::pair<uint64_t, uint64_t> p1, const std::pair<uint64_t, uint64_t> p2)
{
    return p1.first <= p2.first && p1.second <= p2.second;
}

void Utils::GetAllFiles(const std::string &dirName, std::vector<std::string> &files)
{
    struct stat s;
    stat(dirName.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        if (S_ISREG(s.st_mode)) {
            files.push_back(dirName);
        }
        return;
    }
    struct dirent *filename;
    DIR *dir;
    dir = opendir(dirName.c_str());
    if (NULL == dir) {
        return;
    }
    while ((filename = readdir(dir)) != NULL) {
        if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
            continue;
        }
        files.push_back(dirName + std::string("/") + std::string(filename->d_name));
    }
}

bool Utils::EndsWith(const std::string &value, const std::string &ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool Utils::UintCompare(uint64_t num1, uint64_t num2)
{
    return num1 < num2;
}

void Utils::StrSplit(const std::string &str, const std::string &pattern, std::vector<std::string> &vec)
{
    int start, end = -1 * pattern.size();
    do {
        start = end + pattern.size();
        end = str.find(pattern, start);
        vec.push_back(str.substr(start, end - start));
    } while (end != -1);
}

bool Utils::ModelCompare(MxBase::Model *model1, MxBase::Model *model2)
{
    std::vector<std::vector<uint64_t>> dynamicGearInfo1 = model1->GetDynamicGearInfo();
    std::vector<std::vector<uint64_t>> dynamicGearInfo2 = model2->GetDynamicGearInfo();
    return dynamicGearInfo1[0][0] < dynamicGearInfo2[0][0];
}

void Utils::MakeDir(const std::string &path, bool replace)
{
    if (replace && access(path.c_str(), 0) != -1) {
        system(("rm -r " + path).c_str());
        LogInfo << path << " removed!";
    }
    if (access(path.c_str(), 0) == -1) {
        system(("mkdir -p " + path).c_str());
        LogInfo << path << " create!";
    }
}

std::string Utils::GenerateResName(const std::string &basename)
{
    std::string rawName = basename;
    size_t lastIndex = rawName.find_last_of('.');
    rawName = rawName.substr(0, lastIndex);
    size_t underscoreIndex = rawName.find_last_of('_');
    std::string saveName = rawName.substr(underscoreIndex, rawName.length());
    return "infer_img" + saveName + ".txt";
}

std::string Utils::BoolCast(const bool b)
{
    return b ? "true" : "false";
}

void Utils::LoadFromFileVec(const std::string &filename, std::vector<uint64_t> &vec)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    uint64_t num;
    while (file.read(reinterpret_cast<char *>(&num), sizeof(num))) {
        vec.push_back(num);
    }
}

void Utils::SaveToFileVec(const std::string &filename, std::vector<uint64_t> &vec)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    for (auto &inf : vec) {
        file.write(reinterpret_cast<const char *>(&inf), sizeof(inf));
    }
}
