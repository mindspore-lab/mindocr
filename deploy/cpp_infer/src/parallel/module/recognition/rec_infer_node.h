#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_RECOGNITION_REC_INFER_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_RECOGNITION_REC_INFER_NODE_H_

#include <vector>
#include <memory>
#include "postprocess/rec_postprocess.h"
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "data_type/data_type.h"
#include "profile/profile.h"
#include "MxBase/MxBase.h"
#include "Log/Log.h"

class RecInferNode : public AscendBaseModule::ModuleBase {
 public:
  RecInferNode();

  ~RecInferNode() override;

  Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

  Status DeInit() override;

 protected:
  Status Process(std::shared_ptr<void> inputData) override;

 private:
  int stdHeight_ = 48;
  int32_t deviceId_ = 0;
  bool staticMethod_ = true;
  std::vector<MxBase::Model *> crnnNetMindX_;
  std::vector<LiteModelWrap *> crnnNetLite_;
  std::vector<uint32_t> batchSizeList_;
  BackendType backend_;

  Status ParseConfig(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs);

  Status MindXModelInfer(const std::shared_ptr<CommonData> &data);

  Status LiteModelInfer(const std::shared_ptr<CommonData> &data);
};

MODULE_REGIST(RecInferNode)

#endif
