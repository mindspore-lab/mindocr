#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <csignal>
#include <thread>
#include "Log/Log.h"
#include "MxBase/MxBase.h"
#include "utils/utils.h"
#include "profile/profile.h"
#include "detection/det_pre_node.h"
#include "detection/det_infer_node.h"
#include "detection/det_post_node.h"
#include "classification/cls_pre_node.h"
#include "classification/cls_infer_node.h"
#include "classification/cls_post_node.h"
#include "recognition/rec_pre_node.h"
#include "recognition/rec_infer_node.h"
#include "recognition/rec_post_node.h"
#include "common/collect_node.h"
#include "common/hand_out_node.h"
#include "config_parser/config_parser.h"
#include "command_parser/command_parser.h"
#include "framework/module_manager.h"

using AscendBaseModule::MT_HandoutNode;

using AscendBaseModule::MT_DetPreNode;
using AscendBaseModule::MT_DetInferNode;
using AscendBaseModule::MT_DetPostNode;
using AscendBaseModule::MT_ClsPreNode;
using AscendBaseModule::MT_ClsInferNode;
using AscendBaseModule::MT_ClsPostNode;
using AscendBaseModule::MT_RecPreNode;
using AscendBaseModule::MT_RecInferNode;
using AscendBaseModule::MT_RecPostNode;
using AscendBaseModule::MT_CollectNode;
using AscendBaseModule::MODULE_CONNECT_RANDOM;
using AscendBaseModule::ModuleManager;

using std::ios;

namespace {
void SigHandler(int signal) {
  if (signal == SIGINT) {
    Profile::signalReceived_ = true;
  }
}

void ModuleDescGenerator(int deviceNum, std::vector<AscendBaseModule::ModuleDesc> *moduleDesc, TaskType taskType) {
  moduleDesc->push_back({MT_HandoutNode, 1});

  if (taskType == TaskType::DET || taskType == TaskType::DET_REC || taskType == TaskType::DET_CLS_REC) {
    moduleDesc->push_back({MT_DetPreNode, static_cast<int>(std::ceil(0.6 * deviceNum))});
    moduleDesc->push_back({MT_DetInferNode, static_cast<int>(std::ceil(1 * deviceNum))});
    moduleDesc->push_back({MT_DetPostNode, static_cast<int>(std::ceil(2 * deviceNum))});
  }
  if (taskType == TaskType::CLS || taskType == TaskType::DET_CLS_REC) {
    moduleDesc->push_back({MT_ClsPreNode, static_cast<int>(std::ceil(0.6 * deviceNum))});
    moduleDesc->push_back({MT_ClsInferNode, static_cast<int>(std::ceil(1 * deviceNum))});
    moduleDesc->push_back({MT_ClsPostNode, static_cast<int>(std::ceil(0.6 * deviceNum))});
  }
  if (taskType == TaskType::REC || taskType == TaskType::DET_REC || taskType == TaskType::DET_CLS_REC) {
    moduleDesc->push_back({MT_RecPreNode, static_cast<int>(std::ceil(0.7 * deviceNum))});
    moduleDesc->push_back({MT_RecInferNode, static_cast<int>(std::ceil(1 * deviceNum))});
    moduleDesc->push_back({MT_RecPostNode, static_cast<int>(std::ceil(0.26 * deviceNum))});
  }
  moduleDesc->push_back({MT_CollectNode, 1});
}

void ModuleConnectDesc(std::vector<AscendBaseModule::ModuleConnectDesc> *connectDesc, TaskType taskType) {
  // det connect
  std::string preModule = MT_HandoutNode;
  std::string lastModule;

  if (taskType == TaskType::DET || taskType == TaskType::DET_REC || taskType == TaskType::DET_CLS_REC) {
    connectDesc->push_back({preModule, MT_DetPreNode, MODULE_CONNECT_RANDOM});
    connectDesc->push_back({MT_DetPreNode, MT_DetInferNode, MODULE_CONNECT_RANDOM});
    connectDesc->push_back({MT_DetInferNode, MT_DetPostNode, MODULE_CONNECT_RANDOM});
    preModule = MT_DetPostNode;
    lastModule = MT_DetPostNode;
  }
  if (taskType == TaskType::CLS || taskType == TaskType::DET_CLS_REC) {
    connectDesc->push_back({preModule, MT_ClsPreNode, MODULE_CONNECT_RANDOM});
    connectDesc->push_back({MT_ClsPreNode, MT_ClsInferNode, MODULE_CONNECT_RANDOM});
    connectDesc->push_back({MT_ClsInferNode, MT_ClsPostNode, MODULE_CONNECT_RANDOM});
    preModule = MT_ClsPostNode;
    lastModule = MT_ClsPostNode;
  }
  if (taskType == TaskType::REC || taskType == TaskType::DET_REC || taskType == TaskType::DET_CLS_REC) {
    connectDesc->push_back({preModule, MT_RecPreNode, MODULE_CONNECT_RANDOM});
    connectDesc->push_back({MT_RecPreNode, MT_RecInferNode, MODULE_CONNECT_RANDOM});
    connectDesc->push_back({MT_RecInferNode, MT_RecPostNode, MODULE_CONNECT_RANDOM});
    lastModule = MT_RecPostNode;
  }
  connectDesc->push_back({lastModule, MT_CollectNode, MODULE_CONNECT_RANDOM});
}

void DescGenerator(CommandParser *options, std::vector<AscendBaseModule::ModuleConnectDesc> *connectDesc,
                   std::vector<AscendBaseModule::ModuleDesc> *moduleDesc, TaskType taskType) {
  std::vector<uint32_t> deviceIdVec;
  Status ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK) {
    LogError << "Get Device ID failed.";
    exit(-1);
  }
  int deviceNum = static_cast<int>(deviceIdVec.size());

  ModuleDescGenerator(deviceNum, moduleDesc, taskType);

  ModuleConnectDesc(connectDesc, taskType);
}

Status InitModuleManager(ModuleManager *moduleManager, CommandParser *options, const std::string &aclConfigPath,
                         const std::string &pipeline, TaskType taskType) {
  std::vector<AscendBaseModule::ModuleConnectDesc> connectDesc;
  std::vector<AscendBaseModule::ModuleDesc> moduleDesc;
  DescGenerator(options, &connectDesc, &moduleDesc, taskType);

  LogInfo << "ModuleManager: begin to init";
  Status ret = moduleManager->Init(options, aclConfigPath);
  if (ret != Status::OK) {
    LogError << "Fail to init system manager.";
    return Status::COMM_FAILURE;
  }

  ret = moduleManager->RegisterModules(pipeline, moduleDesc.data(), static_cast<int>(moduleDesc.size()), 0);
  if (ret != Status::OK) {
    return Status::COMM_FAILURE;
  }

  ret = moduleManager->RegisterModuleConnects(pipeline, connectDesc.data(), static_cast<int>(connectDesc.size()));
  if (ret != Status::OK) {
    LogError << "Fail to connect module.";
    return Status::COMM_FAILURE;
  }

  return Status::OK;
}

Status DeInitModuleManager(ModuleManager *moduleManager) {
  Status ret = moduleManager->DeInit();
  if (ret != Status::OK) {
    LogError << "Fail to deinit system manager.";
    return Status::COMM_FAILURE;
  }

  return Status::OK;
}

inline void MainAssert(Status exp) {
  if (exp != Status::OK) {
    exit(static_cast<int>(exp));
  }
}

void MainProcess(CommandParser *options, const std::string &streamName, TaskType taskType) {
  LogInfo << "streamName: " << streamName;
  std::string aclConfig;

  std::chrono::high_resolution_clock::time_point endTime;
  std::chrono::high_resolution_clock::time_point startTime;

  ModuleManager moduleManager;
  try {
    MainAssert(InitModuleManager(&moduleManager, options, aclConfig, streamName, taskType));
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
  while (!Profile::signalReceived_) {
    usleep(signalCheckInterval);
  }
  endTime = std::chrono::high_resolution_clock::now();

  double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

  try {
    MainAssert(DeInitModuleManager(&moduleManager));
  } catch (...) {
    LogError << "error occurred during deinit module manager.";
    return;
  }

  LogInfo << "DeInitModuleManager: " << streamName;

  LogInfo << "end to end cost: " << costMs << " in ms / " << costMs / S2MS << " in s. ";
}

int SplitImage(int threadCount, const std::string &imgDir, bool useCls) {
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
    } else if (ptr->d_type == DT_LNK || ptr->d_type == DT_DIR) {
      continue;
    } else {
      std::string filePath = imgDir + "/" + ptr->d_name;
      imgVec.push_back(filePath);
      totalImg++;
    }
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

Status ParseArgs(int argc, const char *argv[], CommandParser *options) {
  LogDebug << "Begin to parse and check command line.";
  options->AddOption("--input_images_dir",
                     "",
                     "Input images dir for inference, can be dir containing multiple images or path of single image.");
  options->AddOption("--det_model_path", "", "Detection model file path.");
  options->AddOption("--cls_model_path", "", "Classification model file path.");
  options->AddOption("--rec_model_path", "", "Recognition model file path or directory.");
  options->AddOption("--character_dict_path", "", "Character dict file path for recognition models.");
  options->AddOption("--backend", "acl", "Inference engine type.");
  options->AddOption("--device", "Ascend", "Device type");
  options->AddOption("--device_type", "310", "310p or 310");
  options->AddOption("--device_id", "0", "Device id.");
  options->AddOption("--parallel_num", "1", "Number of parallel in each stage of pipeline parallelism.");
  options->AddOption("--res_save_dir", "inference_results", "Saving dir for inference results.");
  options->AddOption("--static_rec_model_mode", "true", "true:  static shape, false: dynamic shape");
  options->AddOption("--rec_height", "32", "rec model required height");
  options->AddOption("--rec_min_width", "320", "rec model required min width");
  options->AddOption("--rec_max_width", "2240", "rec model required max width");
  options->ParseArgs(argc, argv);

  int threadCount = options->GetIntOption("--parallel_num");
  if (threadCount < 1) {
    LogError << "The number of threads cannot be less than one.";
    return Status::COMM_INVALID_PARAM;
  }

  std::string inputImgDir = options->GetStringOption("--input_images_dir");
  Status ret = Utils::CheckPath(inputImgDir, "input images path");
  if (ret != Status::OK) {
    LogError << "Parse the dir of input images failed, please check if the path is correct.";
    return ret;
  }

  return Status::OK;
}

void
saveModelGear(std::string *modelPath,
              const int32_t &deviceId,
              const std::string &modelType,
              const BackendType engine) {
  std::vector<std::vector<uint64_t>> dynamicGearInfo;
  if (engine == BackendType::ACL) {
    MxBase::Model model(*modelPath, deviceId);
    dynamicGearInfo = model.GetDynamicGearInfo();
  } else if (engine == BackendType::LITE) {
    dynamicGearInfo = Utils::GetGearInfo(*modelPath);
  } else {
    LogError << "Unsupported engine type";
    return;
  }

  std::vector<std::pair<uint64_t, uint64_t>> gearInfo;
  uint64_t batchInfo;
  for (auto &info : dynamicGearInfo) {
    gearInfo.emplace_back(info[SHAPE_WIDTH_INDEX], info[SHAPE_HEIGHT_INDEX]);
    batchInfo = info[SHAPE_BATCH_SIZE_INDEX];
  }

  std::string baseName = Utils::BaseName(*modelPath) + "." + std::to_string(batchInfo) + ".bin";
  std::string savePath = "./temp/" + modelType + "/";
  Utils::MakeDir(savePath, false);

  Utils::SaveToFilePair(savePath + baseName, gearInfo);
}

void
saveModelBs(std::string *modelPath,
            const int32_t &deviceId,
            const std::string &modelType,
            const BackendType engine) {
  std::vector<std::vector<uint64_t>> dynamicGearInfo;
  if (engine == BackendType::ACL) {
    MxBase::Model model(*modelPath, deviceId);
    dynamicGearInfo = model.GetDynamicGearInfo();
  } else if (engine == BackendType::LITE) {
    dynamicGearInfo = Utils::GetGearInfo(*modelPath);
  } else {
    LogError << "Unsupported engine type";
    return;
  }
  std::vector<uint64_t> batchInfo;
  for (auto &info : dynamicGearInfo) {
    batchInfo.emplace_back(info[0]);
  }

  std::string baseName = Utils::BaseName(*modelPath) + ".bin";
  std::string savePath = "./temp/" + modelType + "/";
  Utils::MakeDir(savePath, false);

  Utils::SaveToFileVec(savePath + baseName, batchInfo);
}

Status configGenerate(CommandParser *options, bool useCls) {
  std::vector<uint32_t> deviceIdVec;
  Status ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK || deviceIdVec.empty()) {
    LogError << "Get Device ID failed.";
    exit(-1);
  }
  int32_t deviceId_ = deviceIdVec[0];

  std::string detModelPath = options->GetStringOption("--det_model_path");
  BackendType backend = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  if (!detModelPath.empty()) {
    saveModelGear(&detModelPath, deviceId_, "dbnet", backend);
  }
  if (useCls) {
    std::string clsModelPath = options->GetStringOption("--cls_model_path");
    saveModelBs(&clsModelPath, deviceId_, "cls", backend);
  }

  bool staticMethod = options->GetBoolOption("--static_rec_model_mode");
  if (staticMethod) {
    std::string recModelPath = options->GetStringOption("--rec_model_path");
    std::vector<std::string> files;
    Utils::GetAllFiles(recModelPath, &files);
    for (auto &file : files) {
      if (Utils::EndsWith(file, ".om")) {
        saveModelGear(&file, deviceId_, "crnn", backend);
      }
    }
  }

  return Status::OK;
}

void PrintProfiling(bool useCls, int totalImgs) {
  LogInfo << "Average Det Inference Time: " << Profile::detInferTime_ / totalImgs << "ms.";
  if (useCls) {
    LogInfo << "Average Cls Inference Time: " << Profile::clsInferTime_ / totalImgs << "ms.";
  }
  LogInfo << "Average Rec Inference Time: " << Profile::recInferTime_ / totalImgs << "ms.";
  LogInfo << "-----------------------------------------------------";

  LogInfo << "Average Det PreProcess Time: " << Profile::detPreProcessTime_ / totalImgs << "ms.";
  LogInfo << "Average Det InferProcess Time: " << Profile::detInferProcessTime_ / totalImgs << "ms.";
  LogInfo << "Average Det PostProcess Time: " << Profile::detPostProcessTime_ / totalImgs << "ms.";
  LogInfo << "-----------------------------------------------------";
  if (useCls) {
    LogInfo << "Average Cls PreProcess Time: " << Profile::clsPreProcessTime_ / totalImgs << "ms.";
    LogInfo << "Average Cls InferProcess Time: " << Profile::clsInferProcessTime_ / totalImgs << "ms.";
    LogInfo << "Average Cls PostProcess Time: " << Profile::clsPostProcessTime_ / totalImgs << "ms.";
    LogInfo << "-----------------------------------------------------";
  }

  LogInfo << "Average Rec PreProcess Time: " << Profile::recPreProcessTime_ / totalImgs << "ms.";
  LogInfo << "Average Rec InferProcess Time: " << Profile::recInferProcessTime_ / totalImgs << "ms.";
  LogInfo << "Average Rec PostProcess Time: " << Profile::recPostProcessTime_ / totalImgs << "ms.";
  LogInfo << "-----------------------------------------------------";
  LogInfo << "Average E2E Process Time: " << Profile::e2eProcessTime_ / totalImgs << "ms.";
}
}  // namespace

Status args_check(CommandParser *options, bool useCls) {
  std::string model_path;

  // device id check
  std::vector<uint32_t> deviceIdVec;
  Status ret = options->GetVectorUint32Value("--device_id", &deviceIdVec);
  if (ret != Status::OK || deviceIdVec.empty()) {
    LogError << "Get device id failed, please check the value of deviceId";
    return Status::COMM_INVALID_PARAM;
  }
  int32_t deviceId;
  for (auto &deviceId_ : deviceIdVec) {
    deviceId = (int32_t) deviceId_;
    if (deviceId_ < MIN_DEVICE_NO || deviceId_ > MAX_DEVICE_NO) {
      LogError << "deviceId must between [0,7]";
      return Status::COMM_INVALID_PARAM;
    }
  }

  BackendType backend = Utils::ConvertBackendTypeToEnum(options->GetStringOption("--backend"));
  if (backend == BackendType::UNSUPPORTED) {
    LogError << "Backend only support acl or lite, please check the value of backend";
    return Status::COMM_INVALID_PARAM;
  }

  // device type check
  std::string deviceType = options->GetStringOption("--device_type");
  if (deviceType != "310P" && deviceType != "310") {
    LogError << "Device type only support 310 or 310P, please check the value of device type.";
    return Status::COMM_INVALID_PARAM;
  }

  std::vector<std::vector<uint64_t>> dynamicGearInfo;

  // det model check
  std::string detModelPath = options->GetStringOption("--det_model_path");
  if (!detModelPath.empty()) {
    ret = Utils::CheckPath(detModelPath, "detModelPath");
    if (ret != Status::OK) {
      LogError << "detModelPath: " << detModelPath << " is not exist or can not read.";
      return Status::COMM_INVALID_PARAM;
    }
    if (backend == BackendType::ACL) {
      try {
        MxBase::Model model(detModelPath, deviceId);
        dynamicGearInfo = model.GetDynamicGearInfo();
      } catch (...) {
        LogError << "please check the value of detModelPath";
        return Status::COMM_INVALID_PARAM;
      }
    } else if (backend == BackendType::LITE) {
      dynamicGearInfo = Utils::GetGearInfo(detModelPath);
    }
    if (dynamicGearInfo.empty()) {
      LogError << "please check the value of detModelPath";
      return Status::COMM_INVALID_PARAM;
    }
  }

  if (useCls) {
    auto clsModelPath = options->GetStringOption("--cls_model_path");
    if (backend == BackendType::ACL) {
      try {
        MxBase::Model model(clsModelPath, deviceId);
        dynamicGearInfo = model.GetDynamicGearInfo();
      } catch (...) {
        LogError << "please check the value of clsModelPath";
        return Status::COMM_INVALID_PARAM;
      }
    } else if (backend == BackendType::LITE) {
      dynamicGearInfo = Utils::GetGearInfo(clsModelPath);
    }
    if (dynamicGearInfo.empty()) {
      LogError << "Please check the value of clsModelPath";
      return Status::COMM_INVALID_PARAM;
    }
  }

  auto staticRecModelMode = options->GetBoolOption("--static_rec_model_mode");
  auto recModelPath = options->GetStringOption("--rec_model_path");
  if (!recModelPath.empty()) {
    ret = Utils::CheckPath(recModelPath, "rec_model_path");
    if (ret != Status::OK) {
      LogError << "rec model path: " << recModelPath << " is not exist of can not read.";
      return Status::COMM_INVALID_PARAM;
    }
    if (!staticRecModelMode) {
      if (backend == BackendType::ACL) {
        try {
          MxBase::Model crnn(recModelPath, deviceId);
        } catch (...) {
          LogError << "please check the value of recModelPath";
          return Status::COMM_INVALID_PARAM;
        }
      }
    }
    std::vector<std::string> files;
    Utils::GetAllFiles(recModelPath, &files);
    for (auto &file : files) {
      try {
        if (backend == BackendType::ACL) {
          MxBase::Model crnn(file, deviceId);
          if (staticRecModelMode) {
            dynamicGearInfo = crnn.GetDynamicGearInfo();
            if (dynamicGearInfo.empty()) {
              LogError << "please check the value of recModelPath";
              return Status::COMM_INVALID_PARAM;
            }
          }
        } else if (backend == BackendType::LITE) {
          if (staticRecModelMode) {
            dynamicGearInfo = Utils::GetGearInfo(file);
            if (dynamicGearInfo.empty()) {
              LogError << "please check the value of recModelPath";
              return Status::COMM_INVALID_PARAM;
            }
          }
        }
      } catch (...) {
        LogError << "please check the value of recModelPath";
        return Status::COMM_INVALID_PARAM;
      }
    }

    if (!staticRecModelMode) {
      int32_t recMinWidth = 0;
      int32_t recMaxWidth = 0;

      auto height = options->GetIntOption("--rec_height");
      if (ret != Status::OK) {
        LogError << "Get recHeight failed, please check the value of recHeight";
        return Status::COMM_INVALID_PARAM;
      }

      if (height < 1) {
        LogError << "recHeight: " << height << " is less than 1, not valid";
        return Status::COMM_INVALID_PARAM;
      }

      recMinWidth = options->GetIntOption("--rec_min_width");
      if (recMinWidth < 1) {
        LogError << "recMinWidth: " << recMinWidth << " is less than 1, not valid";
        return Status::COMM_INVALID_PARAM;
      }

      recMaxWidth = options->GetIntOption("--rec_max_width");
      if (recMaxWidth < 1) {
        LogError << "recMaxWidth: " << recMaxWidth << " is less than 1, not valid";
        return Status::COMM_INVALID_PARAM;
      }

      if (recMaxWidth < recMinWidth) {
        LogError << "recMinWidth must be smaller than recMaxWidth.";
        return Status::COMM_INVALID_PARAM;
      }
    }

    std::string recDictionary = options->GetStringOption("--character_dict_path");

    ret = Utils::CheckPath(recDictionary, "character label file");
    if (ret != Status::OK) {
      LogError << "Character label file: " << recDictionary << " does not exist or cannot be read";
      return Status::COMM_INVALID_PARAM;
    }
  }
  return Status::OK;
}

int main(int argc, const char *argv[]) {
  MxBase::MxInit();
  CommandParser options;
  Status ret = ParseArgs(argc, argv, &options);
  if (ret != Status::OK) {
    LogError << "Parse parameter failed.";
    exit(-1);
  }

  int threadCount = options.GetIntOption("--parallel_num");
  std::string inputImgDir = options.GetStringOption("--input_images_dir");
  std::string clsModelPath = options.GetStringOption("--cls_model_path");
  bool useCls = false;
  if (!clsModelPath.empty()) {
    useCls = true;
  }
  int totalImgs = SplitImage(threadCount, inputImgDir, useCls);
  if (threadCount > totalImgs) {
    LogError << "thread number [" << threadCount << "] can not bigger than total number of input images [" <<
             totalImgs << "].";
    exit(-1);
  }

  if (threadCount < MIN_THREAD_COUNT) {
    LogError << "thread number [" << threadCount << "] cannot be smaller than 1";
    exit(-1);
  }

  if (threadCount > MAX_THREAD_COUNT) {
    LogError << "thread number [" << threadCount << "] cannot be great than 4";
    exit(-1);
  }

  Profile::GetInstance().SetThreadNum(threadCount);

  MainAssert(args_check(&options, useCls));
  ret = configGenerate(&options, useCls);
  if (ret != Status::OK) {
    LogError << "config  generate failed.";
    exit(-1);
  }

  std::thread threadProcess[MAX_THREAD_COUNT];
  std::string streamName[MAX_THREAD_COUNT];

  TaskType taskType = Utils::GetTaskType(&options);
  if (taskType == TaskType::UNSUPPORTED) {
    LogError << "Unsupported task type";
    exit(static_cast<int>(Status::COMM_INVALID_PARAM));
  }

  for (int i = 0; i < threadCount; ++i) {
    streamName[i] = "imgSplitFile" + std::to_string(i) + Utils::BoolCast(useCls);
    threadProcess[i] = std::thread(MainProcess, &options, streamName[i], taskType);
  }

  for (int j = 0; j < threadCount; ++j) {
    threadProcess[j].join();
  }

  std::string modelConfigPath("./temp");
  if (access(modelConfigPath.c_str(), 0) != -1) {
    auto systemResult = system(("rm -r " + modelConfigPath).c_str());
    if (systemResult != 0) {
      LogInfo << modelConfigPath << "remove failed!";
    }
    LogInfo << modelConfigPath << " removed!";
  }
  PrintProfiling(useCls, totalImgs);
  LogInfo << "program End.";
  return 0;
}
