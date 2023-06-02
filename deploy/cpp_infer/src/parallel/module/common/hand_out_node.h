#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_COMMON_HAND_OUT_NODE_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_MODULE_COMMON_HAND_OUT_NODE_H_

#include <string>
#include <memory>
#include "framework/module_manager.h"
#include "config_parser/config_parser.h"
#include "data_type/data_type.h"
#include "utils/utils.h"
#include "profile/profile.h"
#include "Log/Log.h"

class HandoutNode : public AscendBaseModule::ModuleBase {
 public:
    HandoutNode();

    ~HandoutNode() override;

    Status Init(CommandParser *options, const AscendBaseModule::ModuleInitArgs &initArgs) override;

    Status DeInit() override;

 protected:
    Status Process(std::shared_ptr<void> inputData) override;

 private:
    int imgId_ = 0;
    std::string deviceType_;

    Status ParseConfig(CommandParser *options);

    std::string resultPath_;

    std::string nextModule_;

    BackendType backendType_;

    cv::Mat DecodeImgDvpp(std::string imgPath);

    std::unique_ptr<MxBase::ImageProcessor> imageProcessor_{};
};

MODULE_REGIST(HandoutNode)

#endif
