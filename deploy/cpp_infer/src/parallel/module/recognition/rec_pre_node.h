#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_RECOGNITION_REC_PRE_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_RECOGNITION_REC_PRE_NODE_H_

#include <memory>
#include <utility>
#include <vector>
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "utils/utils.h"
#include "data_type/data_type.h"
#include "data_type/constant.h"
#include "profile/profile.h"
#include "Log/Log.h"

class RecPreNode : public AscendBaseModule::ModuleBase {
 public:
  RecPreNode();

  ~RecPreNode() override;

  Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

  Status DeInit() override;

 protected:
  Status Process(std::shared_ptr<void> inputData) override;

 private:
  int stdHeight_ = 48;
  int recMinWidth_ = 320;
  int recMaxWidth_ = 2240;
  bool staticMethod_ = true;
  std::vector<std::pair<uint64_t, uint64_t>> gearInfo_;
  std::vector<uint64_t> batchSizeList_;

  Status ParseConfig(CommandParser *options);

  std::vector<uint32_t> GetCrnnBatchSize(uint32_t frameSize);

  int GetCrnnMaxWidth(const std::vector<cv::Mat> &frames, float maxWHRatio);

  uint8_t *PreprocessCrnn(const std::vector<cv::Mat> &frames, uint32_t batchSize, int maxResizedW, float maxWHRatio,
                          std::vector<ResizedImageInfo> *resizedImageInfos);

  void GetGearInfo(int maxResizedW, std::pair<uint64_t, uint64_t> *gear);

  TaskType taskType_;
};

MODULE_REGIST(RecPreNode)

#endif
