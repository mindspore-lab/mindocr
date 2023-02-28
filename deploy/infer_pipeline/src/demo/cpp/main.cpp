/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Main file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "Utils.h"
#include "Signal.h"
#include "HandOutProcess/HandOutProcess.h"
#include "DbnetPreProcess/DbnetPreProcess.h"
#include "DbnetInferProcess/DbnetInferProcess.h"
#include "DbnetPostProcess/DbnetPostProcess.h"
#include "ClsPreProcess/ClsPreProcess.h"
#include "ClsInferProcess/ClsInferProcess.h"
#include "ClsPostProcess/ClsPostProcess.h"
#include "CrnnPreProcess/CrnnPreProcess.h"
#include "CrnnInferProcess/CrnnInferProcess.h"
#include "CrnnPostProcess/CrnnPostProcess.h"
#include "CollectProcess/CollectProcess.h"

#include "ConfigParser/ConfigParser.h"
#include "CommandParser/CommandParser.h"
#include "ModuleManager/ModuleManager.h"
#include "Log/Log.h"

#include "MxBase/MxBase.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <experimental/filesystem>
#include <csignal>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <dirent.h>

using namespace ascendBaseModule;
using namespace std;

namespace {
void SigHandler(int signal)
{
    if (signal == SIGINT) {
        Signal::signalRecieved = true;
    }
}

void ModuleDescGenerator(int device_num, std::vector<ascendBaseModule::ModuleDesc> &moduleDesc, bool useCls)
{
    moduleDesc.push_back({ MT_HandOutProcess, 1 });
    moduleDesc.push_back({ MT_DbnetPreProcess, static_cast<int>(std::ceil(0.6 * device_num)) });
    moduleDesc.push_back({ MT_DbnetInferProcess, static_cast<int>(std::ceil(1 * device_num)) });
    moduleDesc.push_back({ MT_DbnetPostProcess, static_cast<int>(std::ceil(2 * device_num)) });

    // cls
    if (useCls) {
        moduleDesc.push_back({ MT_ClsPreProcess, static_cast<int>(std::ceil(0.6 * device_num)) });
        moduleDesc.push_back({ MT_ClsInferProcess, static_cast<int>(std::ceil(1 * device_num)) });
        moduleDesc.push_back({ MT_ClsPostProcess, static_cast<int>(std::ceil(0.6 * device_num)) });
    }

    // rec
    moduleDesc.push_back({ MT_CrnnPreProcess, static_cast<int>(std::ceil(0.7 * device_num)) });
    moduleDesc.push_back({ MT_CrnnInferProcess, static_cast<int>(std::ceil(1 * device_num)) });
    moduleDesc.push_back({ MT_CrnnPostProcess, static_cast<int>(std::ceil(0.26 * device_num)) });

    moduleDesc.push_back({ MT_CollectProcess, 1 });
}

void ModuleConnectDesc(std::vector<ascendBaseModule::ModuleConnectDesc> &connectDesc, bool useCls)
{
    // det connect
    connectDesc.push_back({ MT_HandOutProcess, MT_DbnetPreProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_DbnetPreProcess, MT_DbnetInferProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_DbnetInferProcess, MT_DbnetPostProcess, MODULE_CONNECT_RANDOM });
    std::string preModule;

    // cls connect
    if (useCls) {
        connectDesc.push_back({ MT_DbnetPostProcess, MT_ClsPreProcess, MODULE_CONNECT_RANDOM });
        connectDesc.push_back({ MT_ClsPreProcess, MT_ClsInferProcess, MODULE_CONNECT_RANDOM });
        connectDesc.push_back({ MT_ClsInferProcess, MT_ClsPostProcess, MODULE_CONNECT_RANDOM });
        preModule = MT_ClsPostProcess;
    } else {
        preModule = MT_DbnetPostProcess;
    }

    // rec connect
    connectDesc.push_back({ preModule, MT_CrnnPreProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_CrnnPreProcess, MT_CrnnInferProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_CrnnInferProcess, MT_CrnnPostProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_CrnnPostProcess, MT_CollectProcess, MODULE_CONNECT_RANDOM });
}

void DescGenerator(std::string &configPath, std::vector<ascendBaseModule::ModuleConnectDesc> &connectDesc,
    std::vector<ascendBaseModule::ModuleDesc> &moduleDesc, bool useCls)
{
    ConfigParser config;
    config.ParseConfig(configPath);
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = config.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK) {
        LogError << "Get Device ID failed.";
        exit(-1);
    }
    int device_num = (int)deviceIdVec.size();

    ModuleDescGenerator(device_num, moduleDesc, useCls);

    ModuleConnectDesc(connectDesc, useCls);
}

APP_ERROR InitModuleManager(ModuleManager &moduleManager, std::string &configPath, std::string &aclConfigPath,
    const std::string &pipeline, bool useCls)
{
    std::vector<ascendBaseModule::ModuleConnectDesc> connectDesc;
    std::vector<ascendBaseModule::ModuleDesc> moduleDesc;
    DescGenerator(configPath, connectDesc, moduleDesc, useCls);

    LogInfo << "ModuleManager: begin to init";
    APP_ERROR ret = moduleManager.Init(configPath, aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to init system manager, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    ret = moduleManager.RegisterModules(pipeline, moduleDesc.data(), (int)moduleDesc.size(), 0);

    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }

    ret = moduleManager.RegisterModuleConnects(pipeline, connectDesc.data(), (int)connectDesc.size());

    if (ret != APP_ERR_OK) {
        LogError << "Fail to connect module, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}

APP_ERROR DeInitModuleManager(ModuleManager &moduleManager)
{
    APP_ERROR ret = moduleManager.DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Fail to deinit system manager, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}

inline void MainAssert(int exp)
{
    if (exp != APP_ERR_OK) {
        exit(exp);
    }
}

void MainProcess(const std::string &streamName, std::string config, bool useCls)
{
    LogInfo << "streamName: " << streamName;
    std::string aclConfig;

    std::chrono::high_resolution_clock::time_point endTime;
    std::chrono::high_resolution_clock::time_point startTime;

    ModuleManager moduleManager;
    try {
        MainAssert(InitModuleManager(moduleManager, config, aclConfig, streamName, useCls));
    } catch (...) {
        LogError << "error occurred during init module.";
        return;
    }

    startTime = std::chrono::high_resolution_clock::now();
    try {
        MainAssert(moduleManager.RunPipeline());
    } catch (...) {
        LogError << "error occurred during start pipeline.";
        return;
    }

    LogInfo << "wait for exit signal";
    if (signal(SIGINT, SigHandler) == SIG_ERR) {
        LogInfo << "cannot catch SIGINT.";
    }
    const uint16_t signalCheckInterval = 1000;
    while (!Signal::signalRecieved) {
        usleep(signalCheckInterval);
    }
    endTime = std::chrono::high_resolution_clock::now();

    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    try {
        MainAssert(DeInitModuleManager(moduleManager));
    } catch (...) {
        LogError << "error occurred during deinit module manager.";
        return;
    }

    LogInfo << "DeInitModuleManager: " << streamName;

    LogInfo << "end to end cost: " << costMs << " in ms / " << costMs / 1000 << " in s. ";
}

int SplitImage(int threadCount, const std::string &imgDir, bool useCls)
{
    int totalImg = 0;
    DIR *dir = nullptr;
    struct dirent *ptr = nullptr;
    if ((dir = opendir(imgDir.c_str())) == nullptr) {
        LogError << "Open image dir failed, please check the input image dir existed.";
        exit(1);
    }
    std::vector<std::string> imgVec;
    while ((ptr = readdir(dir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == 10 || ptr->d_type == 4) {
            continue;
        } else {
            std::string filePath = imgDir + "/" + ptr->d_name;
            imgVec.push_back(filePath);
            totalImg++;
        };
    }
    closedir(dir);
    sort(imgVec.begin(), imgVec.end());

    std::vector<std::vector<std::string>> fileNum(threadCount);
    for (size_t i = 0; i < imgVec.size(); i++) {
        fileNum[i % threadCount].push_back(imgVec[i]);
    }

    for (int i = 0; i < threadCount; i++) {
        std::ofstream imgFile;
        std::string fileName = "./config/imgSplitFile" + std::to_string(i) + Utils::BoolCast(useCls);
        imgFile.open(fileName, ios::out | ios::trunc);
        for (const auto &img : fileNum[i]) {
            imgFile << img << std::endl;
        }
        imgFile.close();
    }
    LogInfo << "Split Image success.";
    return totalImg;
}

APP_ERROR ParseArgs(int argc, const char *argv[], CommandParser &options)
{
    LogDebug << "Begin to parse and check command line.";
    options.AddOption("-i", "./data/imgDir", "The dir of input images, default: ./data/imgDir");
    options.AddOption("-t", "1", "The number of threads for the program, default: 1");
    options.AddOption("-use_cls", "false", "use cls model, default: false");
    options.AddOption("-config", "./config/setup.config", "The path of config file.");
    options.ParseArgs(argc, argv);

    int threadCount = options.GetIntOption("-t");
    if (threadCount < 1) {
        LogError << "The number of threads cannot be less than one.";
        return APP_ERR_COMM_FAILURE;
    }

    std::string inputImgDir = options.GetStringOption("-i");
    APP_ERROR ret = Utils::CheckPath(inputImgDir, "input images path");
    if (ret != APP_ERR_OK) {
        LogError << "Parse the dir of input images failed, please check if the path is correct.";
        return ret;
    }

    std::string configFile = options.GetStringOption("-config");
    ret = Utils::CheckPath(configFile, "config file path");
    if (ret != APP_ERR_OK) {
        LogError << "Parse the config file path failed, please check if the path is correct.";
        return ret;
    }

    return APP_ERR_OK;
}


void saveModelGear(std::string modelPath, int32_t &deviceId, const std::string &modelType)
{
    MxBase::Model model(modelPath, deviceId);
    std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
    std::vector<std::pair<uint64_t, uint64_t>> gearInfo;
    uint64_t batchInfo;
    for (auto &info : dynamicGearInfo) {
        gearInfo.emplace_back(info[2], info[3]);
        batchInfo = info[0];
    }

    std::string baseName = Utils::BaseName(modelPath) + "." + std::to_string(batchInfo) + ".bin";
    std::string savePath = "./temp/" + modelType + "/";
    Utils::MakeDir(savePath, false);

    Utils::SaveToFilePair(savePath + baseName, gearInfo);
}

void saveModelBs(std::string modelPath, int32_t &deviceId, const std::string &modelType)
{
    MxBase::Model model(modelPath, deviceId);
    std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
    std::vector<uint64_t> batchInfo;
    for (auto &info : dynamicGearInfo) {
        batchInfo.emplace_back(info[0]);
    }

    std::string baseName = Utils::BaseName(modelPath) + ".bin";
    std::string savePath = "./temp/" + modelType + "/";
    Utils::MakeDir(savePath, false);

    Utils::SaveToFileVec(savePath + baseName, batchInfo);
}

APP_ERROR configGenerate(std::string &configPath, bool useCls)
{
    std::string modelConfigPath("./temp");
    Utils::MakeDir(modelConfigPath, true);
    ConfigParser config;
    config.ParseConfig(configPath);

    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = config.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get Device ID failed.";
        exit(-1);
    }
    int32_t deviceId_ = deviceIdVec[0];

    std::string detModelPath;
    ret = config.GetStringValue("detModelPath", detModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Parse the config file path failed, please check if the path is correct.";
        return ret;
    }

    saveModelGear(detModelPath, deviceId_, "dbnet");
    if (useCls) {
        std::string clsModelPath;
        ret = config.GetStringValue("clsModelPath", clsModelPath);
        if (ret != APP_ERR_OK) {
            LogError << "Parse the config file path failed, please check if the path is correct.";
            return ret;
        }
        saveModelBs(clsModelPath, deviceId_, "cls");
    }


    bool staticMethod = true;
    ret = config.GetBoolValue("staticRecModelMode", staticMethod);
    if (ret != APP_ERR_OK) {
        LogError << "Get staticRecModelMode failed, please check the value of staticRecModelMode";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (staticMethod) {
        std::string recModelPath;
        ret = config.GetStringValue("recModelPath", recModelPath);
        if (ret != APP_ERR_OK) {
            LogError << "Get recModelPath failed, please check the value of recModelPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }

        std::vector<std::string> files;
        Utils::GetAllFiles(recModelPath, files);
        for (const auto &file : files) {
            if (Utils::EndsWith(file, ".om")) {
                saveModelGear(file, deviceId_, "crnn");
            }
        }
    }

    return APP_ERR_OK;
}

void PrintProfiling(bool useCls, int totalImgs)
{
    LogInfo << "Average Det Inference Time: " << Signal::detInferTime / totalImgs << "ms.";
    if (useCls) {
        LogInfo << "Average Cls Inference Time: " << Signal::clsInferTime / totalImgs << "ms.";
    }
    LogInfo << "Average Rec Inference Time: " << Signal::recInferTime / totalImgs << "ms.";
    LogInfo << "-----------------------------------------------------";

    LogInfo << "Average Det PreProcess Time: " << Signal::detPreProcessTime / totalImgs << "ms.";
    LogInfo << "Average Det InferProcess Time: " << Signal::detInferProcessTime / totalImgs << "ms.";
    LogInfo << "Average Det PostProcess Time: " << Signal::detPostProcessTime / totalImgs << "ms.";
    LogInfo << "-----------------------------------------------------";
    if (useCls) {
        LogInfo << "Average Cls PreProcess Time: " << Signal::clsPreProcessTime / totalImgs << "ms.";
        LogInfo << "Average Cls InferProcess Time: " << Signal::clsInferProcessTime / totalImgs << "ms.";
        LogInfo << "Average Cls PostProcess Time: " << Signal::clsPostProcessTime / totalImgs << "ms.";
        LogInfo << "-----------------------------------------------------";
    }

    LogInfo << "Average Rec PreProcess Time: " << Signal::recPreProcessTime / totalImgs << "ms.";
    LogInfo << "Average Rec InferProcess Time: " << Signal::recInferProcessTime / totalImgs << "ms.";
    LogInfo << "Average Rec PostProcess Time: " << Signal::recPostProcessTime / totalImgs << "ms.";
    LogInfo << "-----------------------------------------------------";
    LogInfo << "Average E2E Process Time: " << Signal::e2eProcessTime / totalImgs << "ms.";
}
}

APP_ERROR args_check(const std::string &configPath, bool useCls)
{
    ConfigParser configParser;
    configParser.ParseConfig(configPath);
    std::string model_path;

    // device id check
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = configParser.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get device id failed, please check the value of deviceId";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    int32_t deviceId;
    for (auto &deviceId_ : deviceIdVec) {
        deviceId = (int32_t)deviceId_;
        if (deviceId_ < 0 || deviceId_ > 7) {
            LogError << "deviceId must between [0,7]";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }

    // device type check
    std::string deviceType;
    ret = configParser.GetStringValue("deviceType", deviceType);
    if (ret != APP_ERR_OK) {
        LogError << "Get device type failed, please check the value of deviceType";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (deviceType != "310P" && deviceType != "310") {
        LogError << "Device type only support 310 or 310P, please check the value of device type.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    // det model check
    std::string detModelPath;
    ret = configParser.GetStringValue("detModelPath", detModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get detModelPath failed, please check the value of detModelPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ret = Utils::CheckPath(detModelPath, "detModelPath");
    if (ret != APP_ERR_OK) {
        LogError << "detModelPath: " << detModelPath << " is not exist or can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    try {
        MxBase::Model model(detModelPath, deviceId);
        std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
        if (dynamicGearInfo.empty()) {
            LogError << "please check the value of detModelPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    } catch (...) {
        LogError << "please check the value of detModelPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (useCls) {
        ret = configParser.GetStringValue("clsModelPath", model_path);
        if (ret != APP_ERR_OK) {
            LogError << "Parse the config file path failed, please check if the path is correct.";
            return ret;
        }

        try {
            MxBase::Model model(model_path, deviceId);
            std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
            if (dynamicGearInfo.empty()) {
                LogError << "please check the value of clsModelPath";
                return APP_ERR_COMM_INVALID_PARAM;
            }
        } catch (...) {
            LogError << "please check the value of clsModelPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }

    bool staticMethod;
    ret = configParser.GetBoolValue("staticRecModelMode", staticMethod);
    if (ret != APP_ERR_OK) {
        LogError << "Get staticRecModelMode failed, please check the value of staticRecModelMode";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::string recModelPath;
    ret = configParser.GetStringValue("recModelPath", recModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get recModelPath failed, please check the value of recModelPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ret = Utils::CheckPath(recModelPath, "recModelPath");
    if (ret != APP_ERR_OK) {
        LogError << "rec model path: " << recModelPath << " is not exist of can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (!staticMethod) {
        try {
            MxBase::Model crnn(recModelPath, deviceId);
        } catch (...) {
            LogError << "please check the value of recModelPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }

    std::vector<std::string> files;
    Utils::GetAllFiles(recModelPath, files);
    for (auto &file : files) {
        try {
            MxBase::Model crnn(file, deviceId);
            if (staticMethod) {
                std::vector<std::vector<uint64_t>> dynamicGearInfo = crnn.GetDynamicGearInfo();
                if (dynamicGearInfo.empty()) {
                    LogError << "please check the value of recModelPath";
                    return APP_ERR_COMM_INVALID_PARAM;
                }
            }
        } catch (...) {
            LogError << "please check the value of recModelPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }

    if (!staticMethod) {
        int32_t recMinWidth = 0;
        int32_t recMaxWidth = 0;
        int32_t Height = 0;

        ret = configParser.GetIntValue("recHeight", Height);
        if (ret != APP_ERR_OK) {
            LogError << "Get recHeight failed, please check the value of recHeight";
            return APP_ERR_COMM_INVALID_PARAM;
        }

        if (Height < 1) {
            LogError << "recHeight: " << Height << " is less than 1, not valid";
            return APP_ERR_COMM_INVALID_PARAM;
        }

        ret = configParser.GetIntValue("recMinWidth", recMinWidth);

        if (ret != APP_ERR_OK) {
            LogError << "Get recMinWidth failed, please check the value of recMinWidth";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        if (recMinWidth < 1) {
            LogError << "recMinWidth: " << recMinWidth << " is less than 1, not valid";
            return APP_ERR_COMM_INVALID_PARAM;
        }

        ret = configParser.GetIntValue("recMaxWidth", recMaxWidth);
        if (ret != APP_ERR_OK) {
            LogError << "Get recMaxWidth failed, please check the value of recMaxWidth";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        if (recMaxWidth < 1) {
            LogError << "recMaxWidth: " << recMaxWidth << " is less than 1, not valid";
            return APP_ERR_COMM_INVALID_PARAM;
        }

        if (recMaxWidth < recMinWidth) {
            LogError << "recMinWidth must be smaller than recMaxWidth.";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }

    std::string recDictionary;
    ret = configParser.GetStringValue("dictPath", recDictionary);
    if (ret != APP_ERR_OK) {
        LogError << "Get dictPath failed, please check the value of dictPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(recDictionary, "character label file");
    if (ret != APP_ERR_OK) {
        LogError << "Character label file: " << recDictionary << " does not exist or cannot be read";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    bool saveInferResult;
    ret = configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (ret != APP_ERR_OK) {
        LogError << "Get saveInferResult failed, please check the value of saveInferResult";
        return APP_ERR_COMM_INVALID_PARAM;
    }


    return APP_ERR_OK;
}

int main(int argc, const char *argv[])
{
    MxBase::MxInit();
    CommandParser options;
    APP_ERROR ret = ParseArgs(argc, argv, options);
    if (ret != APP_ERR_OK) {
        LogError << "Parse parameter failed.";
        exit(-1);
    }

    int threadCount = options.GetIntOption("-t");
    std::string inputImgDir = options.GetStringOption("-i");
    bool useCls = options.GetBoolOption("-use_cls");
    int totalImgs = SplitImage(threadCount, inputImgDir, useCls);
    if (threadCount > totalImgs) {
        LogError << "thread number [" << threadCount << "] can not bigger than total number of input images [" <<
            totalImgs << "].";
        exit(-1);
    }

    if (threadCount < 1) {
        LogError << "thread number [" << threadCount << "] cannot be smaller than 1";
        exit(-1);
    }

    if (threadCount > 4) {
        LogError << "thread number [" << threadCount << "] cannot be great than 4";
        exit(-1);
    }

    Signal::GetInstance().SetThreadNum(threadCount);

    std::string setupConfig = options.GetStringOption("-config");
    MainAssert(args_check(setupConfig, useCls));

    ret = configGenerate(setupConfig, useCls);
    if (ret != APP_ERR_OK) {
        LogError << "config set up failed.";
        exit(-1);
    }

    std::thread threadProcess[threadCount];
    std::string streamName[threadCount];

    for (int i = 0; i < threadCount; ++i) {
        streamName[i] = "imgSplitFile" + std::to_string(i) + Utils::BoolCast(useCls);
        threadProcess[i] = std::thread(MainProcess, streamName[i], setupConfig, useCls);
    }

    for (int j = 0; j < threadCount; ++j) {
        threadProcess[j].join();
    }

    std::string modelConfigPath("./temp");
    if (access(modelConfigPath.c_str(), 0) != -1) {
        system(("rm -r " + modelConfigPath).c_str());
        LogInfo << modelConfigPath << " removed!";
    }
    PrintProfiling(useCls, totalImgs);
    LogInfo << "program End.";
    return 0;
}