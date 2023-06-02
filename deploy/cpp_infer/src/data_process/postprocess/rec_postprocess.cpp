#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include "Log/Log.h"
#include "postprocess/rec_postprocess.h"

RecCTCLabelDecode::RecCTCLabelDecode() = default;

// trim from end of string (right)
inline std::string &rtrim(std::string *s) {
  const char *ws = "\n\r";
  s->erase(s->find_last_not_of(ws) + 1);
  return *s;
}

void RecCTCLabelDecode::ClassNameInit(const std::string &fileName) {
  std::ifstream fsIn(fileName);
  std::string line;

  if (fsIn) {
    labelVec_.clear();
    labelVec_.emplace_back("");
    while (getline(fsIn, line)) {
      labelVec_.push_back(rtrim(&line));
    }
  } else {
    LogInfo << "no such file";
  }
  labelVec_.emplace_back(" ");
  fsIn.close();

  classNum_ = labelVec_.size();

  LogInfo << " ClassNameInit classNum_ " << classNum_;
}

std::string RecCTCLabelDecode::GetClassName(const size_t classId) {
  if (classId >= labelVec_.size()) {
    LogError << "Failed to get classId(" << classId << ") label, size(" << labelVec_.size() << ").";
    return "";
  }
  return labelVec_[classId];
}

void
RecCTCLabelDecode::CalcMindXOutputIndex(const int64_t *resHostBuf, size_t batchSize, size_t objectNum,
                               std::vector<std::string> *resVec) {
  LogDebug << "Start to Process CalcMindXOutputIndex.";
  batchSize_ = batchSize;
  objectNum_ = objectNum;

  for (uint32_t i = 0; i < batchSize_; i++) {
    int64_t blankIdx = blankIdx_;
    int64_t previousIdx = blankIdx_;
    std::string result;
    std::string str;
    for (int64_t j = 0; j < objectNum_; j++) {
      int64_t outputNum = i * objectNum_ + j;
      if (resHostBuf[outputNum] != blankIdx && resHostBuf[outputNum] != previousIdx) {
        result = GetClassName(resHostBuf[outputNum]);
        str += result;
      }
      previousIdx = resHostBuf[outputNum];
    }
    if (std::strcmp(str.c_str(), "") == 0) {
      str = "###";
    }
    (*resVec)[i] += str;
  }
}

void RecCTCLabelDecode::CalcLiteOutputIndex(const int32_t *resHostBuf, size_t batchSize, size_t objectNum,
                                   std::vector<std::string> *resVec) {
  LogDebug << "Start to Process CalcMindXOutputIndex.";

  batchSize_ = batchSize;
  objectNum_ = objectNum;

  for (uint32_t i = 0; i < batchSize_; i++) {
    int64_t blankIdx = blankIdx_;
    int64_t previousIdx = blankIdx_;
    std::string result;
    std::string str;
    for (int64_t j = 0; j < objectNum_; j++) {
      int64_t outputNum = i * objectNum_ + j;
      if (resHostBuf[outputNum] != blankIdx && resHostBuf[outputNum] != previousIdx) {
        result = GetClassName(resHostBuf[outputNum]);
        str += result;
      }
      previousIdx = resHostBuf[outputNum];
    }
    if (std::strcmp(str.c_str(), "") == 0) {
      str = "###";
    }
    (*resVec)[i] += str;
  }
}
