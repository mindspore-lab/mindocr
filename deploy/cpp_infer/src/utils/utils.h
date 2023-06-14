#ifndef DEPLOY_CPP_INFER_SRC_UTILS_UTILS_H_
#define DEPLOY_CPP_INFER_SRC_UTILS_UTILS_H_

#include <unistd.h>
#include <dirent.h>
#include <utility>
#include <iostream>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"
#include "status_code/status_code.h"
#include "data_type/data_type.h"
#include "command_parser/command_parser.h"
#include "config_parser/config_parser.h"

class Utils {
 public:
  static uint32_t ImageChanSizeF32(uint32_t width, uint32_t height);

  static uint32_t RgbImageSizeF32(uint32_t width, uint32_t height);

  static uint8_t *ImageNchw(const std::vector<cv::Mat> &nhwcImageChs, uint32_t size);

  static Status CheckPath(const std::string &srcPath, const std::string &config);

  static std::string BaseName(const std::string &filename);

  static void LoadFromFilePair(const std::string &filename, std::vector<std::pair<uint64_t, uint64_t>> *vec);

  static void SaveToFilePair(const std::string &filename, const std::vector<std::pair<uint64_t, uint64_t>> &vec);

  static void LoadFromFileVec(const std::string &filename, std::vector<uint64_t> *vec);

  static void SaveToFileVec(const std::string &filename, const std::vector<uint64_t> &vec);

  static bool PairCompare(std::pair<uint64_t, uint64_t> p1, std::pair<uint64_t, uint64_t> p2);

  static bool GearCompare(std::pair<uint64_t, uint64_t> p1, std::pair<uint64_t, uint64_t> p2);

  static bool UintCompare(uint64_t num1, uint64_t num2);

  static bool ModelCompare(MxBase::Model *model1, MxBase::Model *model2);

  static bool LiteModelCompare(LiteModelWrap *model1, LiteModelWrap *model2);

  static void GetAllFiles(const std::string &dirName, std::vector<std::string> *files);

  static bool EndsWith(std::string const &value, std::string const &ending);

  static void StrSplit(const std::string &str, const std::string &pattern, std::vector<std::string> *vec);

  static void MakeDir(const std::string &path, bool replace);

  static std::string GetImageName(const std::string &basename);

  static std::string BoolCast(bool b);

  static BackendType ConvertBackendTypeToEnum(const std::string &engine);

  static std::vector<std::vector<uint64_t>> GetGearInfo(const std::string &file);

  static std::vector<std::string> SplitString(const std::string &basicString, char delim);

  static TaskType GetTaskType(CommandParser *options);
};

#endif
