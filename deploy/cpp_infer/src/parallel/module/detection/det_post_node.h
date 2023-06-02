#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_DETECTION_DET_POST_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_DETECTION_DET_POST_NODE_H_
#include <memory>
#include <string>
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "data_type/data_type.h"
#include "profile/profile.h"
#include "Log/Log.h"
#include "utils/utils.h"

class DetPostNode : public AscendBaseModule::ModuleBase {
 public:
  DetPostNode();

  ~DetPostNode() override;

  Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

  Status DeInit() override;

 protected:
  Status Process(std::shared_ptr<void> inputData) override;

 private:
  std::string resultPath_;
  std::string nextModule_;

  Status ParseConfig(CommandParser *options);

  static float CalcCropWidth(const TextObjectInfo &textObject);

  static float CalcCropHeight(const TextObjectInfo &textObject);
};

MODULE_REGIST(DetPostNode)

#endif
