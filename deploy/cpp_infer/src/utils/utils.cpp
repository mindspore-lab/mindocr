#include <securec.h>
#include <sys/stat.h>
#include <utility>
#include "data_type/constant.h"
#include "data_type/data_type.h"
#include "utils/utils.h"

const int ASCII_START = 32;
const int ASCII_END = 127;
const int SHAPE_POSITION_BIAS = 3;

uint32_t Utils::ImageChanSizeF32(uint32_t width, uint32_t height) {
  return width * height * PIX_BYTES;
}

uint32_t Utils::RgbImageSizeF32(uint32_t width, uint32_t height) {
  return width * height * CHANNEL_SIZE * PIX_BYTES;
}

uint8_t *Utils::ImageNchw(const std::vector<cv::Mat> &nhwcImageChs, uint32_t size) {
  uint8_t *nchwBuf = new uint8_t[size];
  uint32_t channelSize = ImageChanSizeF32(nhwcImageChs[0].rows, nhwcImageChs[0].cols);
  int pos = 0;
  for (unsigned int i = 0; i < nhwcImageChs.size(); i++) {
    if (memcpy_s(static_cast<uint8_t *>(nchwBuf) + pos, channelSize, nhwcImageChs[i].ptr<float>(0), channelSize) ==
        0) {
      pos += channelSize;
    } else {
      LogError << "memcpy_s failed";
      break;
    }
  }
  return nchwBuf;
}

Status Utils::CheckPath(const std::string &srcPath, const std::string &config) {
  int folderExist = access(srcPath.c_str(), R_OK);
  if (folderExist == -1) {
    LogError << config << " doesn't exist or can not read.";
    return Status::COMM_INVALID_PARAM;
  }

  return Status::OK;
}

std::string Utils::BaseName(const std::string &filename) {
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

void Utils::LoadFromFilePair(const std::string &filename, std::vector<std::pair<uint64_t, uint64_t>> *vec) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  std::pair<uint64_t, uint64_t> pair;
  while (file.read(reinterpret_cast<char *>(&pair), sizeof(pair))) {
    vec->push_back(pair);
  }
}

void Utils::SaveToFilePair(const std::string &filename, const std::vector<std::pair<uint64_t, uint64_t>> &vec) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  for (auto &inf : vec) {
    file.write(reinterpret_cast<const char *>(&inf), sizeof(inf));
  }
}

bool Utils::PairCompare(const std::pair<uint64_t, uint64_t> p1, const std::pair<uint64_t, uint64_t> p2) {
  if (p1.first < p2.first) {
    return true;
  } else if (p1.first == p2.first) {
    return p1.second < p2.second;
  }
  return false;
}

bool Utils::GearCompare(const std::pair<uint64_t, uint64_t> p1, const std::pair<uint64_t, uint64_t> p2) {
  return p1.first <= p2.first && p1.second <= p2.second;
}

void Utils::GetAllFiles(const std::string &dirName, std::vector<std::string> *files) {
  struct stat s{};
  stat(dirName.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    if (S_ISREG(s.st_mode)) {
      files->push_back(dirName);
    }
    return;
  }
  struct dirent *filename;
  DIR *dir = opendir(dirName.c_str());
  if (nullptr == dir) {
    return;
  }
  while ((filename = readdir(dir)) != nullptr) {
    if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
      continue;
    }
    files->push_back(dirName + std::string("/") + std::string(filename->d_name));
  }
}

bool Utils::EndsWith(const std::string &value, const std::string &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool Utils::UintCompare(uint64_t num1, uint64_t num2) {
  return num1 < num2;
}

void Utils::StrSplit(const std::string &str, const std::string &pattern, std::vector<std::string> *vec) {
  int start;
  int end = -1 * pattern.size();
  do {
    start = end + pattern.size();
    end = str.find(pattern, start);
    vec->push_back(str.substr(start, end - start));
  } while (end != -1);
}

bool Utils::ModelCompare(MxBase::Model *model1, MxBase::Model *model2) {
  std::vector<std::vector<uint64_t>> dynamicGearInfo1 = model1->GetDynamicGearInfo();
  std::vector<std::vector<uint64_t>> dynamicGearInfo2 = model2->GetDynamicGearInfo();
  return dynamicGearInfo1[0][0] < dynamicGearInfo2[0][0];
}

bool Utils::LiteModelCompare(LiteModelWrap *model1, LiteModelWrap *model2) {
  std::vector<std::vector<uint64_t>> dynamicGearInfo1 = model1->dynamicGearInfo;
  std::vector<std::vector<uint64_t>> dynamicGearInfo2 = model2->dynamicGearInfo;
  return dynamicGearInfo1[0][0] < dynamicGearInfo2[0][0];
}

void Utils::MakeDir(const std::string &path, bool replace) {
  if (replace && access(path.c_str(), 0) != -1) {
    auto ret = system(("rm -r " + path).c_str());
    if (ret != 0) {
      LogError << "Remove path " << path << "failed";
    }
    LogInfo << path << " removed!";
  }
  if (access(path.c_str(), 0) == -1) {
    auto ret = system(("mkdir -p " + path).c_str());
    if (ret != 0) {
      LogError << "Create path " << path << "failed";
    }
    LogInfo << path << " create!";
  }
}

std::string Utils::GetImageName(const std::string &basename) {
  std::string rawName = basename;
  size_t lastIndex = rawName.find_last_of('/');
  rawName = rawName.substr(lastIndex + 1);
  return rawName;
}

std::string Utils::BoolCast(const bool b) {
  return b ? "true" : "false";
}

void Utils::LoadFromFileVec(const std::string &filename, std::vector<uint64_t> *vec) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  uint64_t num;
  while (file.read(reinterpret_cast<char *>(&num), sizeof(num))) {
    vec->push_back(num);
  }
}

void Utils::SaveToFileVec(const std::string &filename, const std::vector<uint64_t> &vec) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  for (auto &inf : vec) {
    file.write(reinterpret_cast<const char *>(&inf), sizeof(inf));
  }
}

BackendType Utils::ConvertBackendTypeToEnum(const std::string &engine) {
  if (engine == "acl") {
    return BackendType::ACL;
  } else if (engine == "lite") {
    return BackendType::LITE;
  } else {
    return BackendType::UNSUPPORTED;
  }
}

std::vector<std::vector<uint64_t>> Utils::GetGearInfo(const std::string &modelPath) {
  std::vector<std::vector<uint64_t>> gears;
  std::ifstream infile(modelPath, std::ios::binary);

  infile.seekg(0, std::ios::end);
  size_t fileSize = infile.tellg();
  infile.seekg(0, std::ios::beg);

  char *buffer = new char[fileSize];

  infile.read(buffer, fileSize);

  infile.close();

  std::string content;
  for (size_t i = 0; i < fileSize; i++) {
    if (buffer[i] >= ASCII_START && buffer[i] <= ASCII_END) {
      content += buffer[i];
    }
  }

  delete[] buffer;

  std::regex pattern("_all_origin_gears_inputs.*?R9_ge_attr");
  std::smatch match;

  if (std::regex_search(content, match, pattern)) {
    std::string matchText = match[0].str();

    std::smatch results;
    std::string temp;
    std::regex shapePattern(":4:(\\d+),(\\d+),(\\d+),(\\d+)");
    std::sregex_iterator it(matchText.begin(), matchText.end(), shapePattern);
    std::sregex_iterator end;

    while (it != end) {
      temp = it->str().substr(SHAPE_POSITION_BIAS);
      std::vector<std::string> shapeVec = Utils::SplitString(temp, ',');
      std::vector<uint64_t> gear;
      for (const auto &s : shapeVec) {
        gear.push_back(std::stoi(s));
      }
      gears.push_back(gear);
      ++it;
    }
  }
  return gears;
}

std::vector<std::string> Utils::SplitString(const std::string &input, const char delim) {
  std::stringstream ss(input);
  std::string item;
  std::vector<std::string> tokens;
  while (getline(ss, item, delim)) {
    tokens.push_back(item);
  }
  return tokens;
}

TaskType Utils::GetTaskType(CommandParser *options) {
  auto det = !options->GetStringOption("--det_model_path").empty();
  auto cls = !options->GetStringOption("--cls_model_path").empty();
  auto rec = !options->GetStringOption("--rec_model_path").empty();
  if (det && !cls && !rec) {
    return TaskType::DET;
  } else if (!det && cls && !rec) {
    return TaskType::CLS;
  } else if (!det && !cls && rec) {
    return TaskType::UNSUPPORTED;
  } else if (det && !cls && rec) {
    return TaskType::DET_REC;
  } else if (det && cls && rec) {
    return TaskType::DET_CLS_REC;
  } else {
    return TaskType::UNSUPPORTED;
  }
}

