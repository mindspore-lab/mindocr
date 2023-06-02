#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_RECOGNITION_REC_POST_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_RECOGNITION_REC_POST_NODE_H_

#include <unordered_set>
#include <memory>
#include <string>
#include <vector>
#include "postprocess/rec_postprocess.h"
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "profile/profile.h"
#include "data_type/data_type.h"
#include "MxBase/MxBase.h"
#include "Log/Log.h"

class RecPostNode : public AscendBaseModule::ModuleBase {
 public:
  RecPostNode();

  ~RecPostNode() override;

  Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

  Status DeInit() override;

 protected:
  Status Process(std::shared_ptr<void> inputData) override;

 private:
  RecCTCLabelDecode recCtcLabelDecode_;
  std::string recDictionary_;
  std::string resultPath_;

  std::unordered_set<int> idSet_;

  Status ParseConfig(CommandParser *options);

  Status PostProcessMindXCrnn(uint32_t framesSize, const std::vector<MxBase::Tensor> &inferOutput,
                              std::vector<std::string> *textsInfos);

  Status PostProcessLiteCrnn(uint32_t framesSize, const std::vector<mindspore::MSTensor> &inferOutput,
                             std::vector<std::string> *textsInfos);
};

MODULE_REGIST(RecPostNode)

#endif
