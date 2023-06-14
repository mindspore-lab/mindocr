#ifndef DEPLOY_CPP_INFER_SRC_DATA_PROCESS_POSTPROCESS_REC_POSTPROCESS_H_
#define DEPLOY_CPP_INFER_SRC_DATA_PROCESS_POSTPROCESS_REC_POSTPROCESS_H_
#include <string>
#include <algorithm>
#include <vector>
#include <memory>
#include "utils/utils.h"

class RecCTCLabelDecode {
 public:
  RecCTCLabelDecode();

  ~RecCTCLabelDecode() = default;

  void ClassNameInit(const std::string &fileName);

  std::string GetClassName(size_t classId);

  void CalcMindXOutputIndex(const int64_t *resHostBuf,
                            size_t batchSize,
                            size_t objectNum,
                            std::vector<std::string> *resVec);

  void CalcLiteOutputIndex(const int32_t *resHostBuf,
                           size_t batchSize,
                           size_t objectNum,
                           std::vector<std::string> *resVec);

 private:
  std::vector<std::string> labelVec_ = {};  // labels info
  uint32_t classNum_ = 0;
  uint32_t objectNum_ = 200;
  uint32_t batchSize_ = 0;
  uint32_t blankIdx_ = 0;
};

#endif
