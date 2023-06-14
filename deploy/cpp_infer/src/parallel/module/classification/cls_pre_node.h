#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_CLASSIFICATION_CLS_PRE_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_CLASSIFICATION_CLS_PRE_NODE_H_

#include <securec.h>
#include <memory>
#include <vector>
#include "Log/Log.h"
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "utils/utils.h"
#include "data_type/data_type.h"
#include "profile/profile.h"
#include "command_parser/command_parser.h"


class ClsPreNode : public AscendBaseModule::ModuleBase {
 public:
    ClsPreNode();

    ~ClsPreNode() override;

    Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

    Status DeInit() override;

 protected:
    Status Process(std::shared_ptr<void> inputData) override;

 private:
    int clsHeight_ = 48;
    int clsWidth_ = 192;
    int32_t deviceId_ = 0;
    std::vector<uint64_t> batchSizeList_;

    Status ParseConfig(CommandParser *options);

    std::vector<uint32_t> GetClsBatchSize(uint32_t frameSize);

    uint8_t *PreprocessCls(const std::vector<cv::Mat> &frames, uint32_t batchSize);

    TaskType taskType_;
};

MODULE_REGIST(ClsPreNode)

#endif
