#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_COMMON_COLLECT_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_COMMON_COLLECT_NODE_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "profile/profile.h"
#include "data_type/data_type.h"
#include "Log/Log.h"

class CollectNode : public AscendBaseModule::ModuleBase {
 public:
  CollectNode();

  ~CollectNode() override;

  Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

  Status DeInit() override;

 protected:
  Status Process(std::shared_ptr<void> inputData) override;

 private:
  std::string resultPath_;
  std::unordered_map<int, int> idMap_;
  int inferSize_ = 0;
  TaskType taskType_;
  std::string saveFileName_;

  Status ParseConfig(CommandParser *options);

  void SignalSend(int imgTotal);

  static std::string GenerateDetClsRecInferRes(const std::string &imageName, uint32_t frameSize,
                                               const std::vector<std::string> &inferRes);

  static std::string GenerateDetInferRes(const std::string &imageName, const std::vector<std::string> &inferRes);

  static std::string GenerateClsInferRes(const std::string &imageName, const std::vector<std::string> &inferRes);

  static std::string GenerateRecInferRes(const std::string &imageName, const std::vector<std::string> &inferRes);
};

MODULE_REGIST(CollectNode)

#endif
