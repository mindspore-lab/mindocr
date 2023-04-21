#include "profile/profile.h"

static std::mutex gMtx = {};
bool Profile::signalReceived_ = false;
double Profile::detInferTime_ = 0;
double Profile::recInferTime_ = 0;
double Profile::e2eProcessTime_ = 0;

double Profile::detPreProcessTime_ = 0;
double Profile::detPostProcessTime_ = 0;

double Profile::clsPreProcessTime_ = 0;
double Profile::clsInferTime_ = 0;
double Profile::clsPostProcessTime_ = 0;

double Profile::recPreProcessTime_ = 0;
double Profile::recPostProcessTime_ = 0;

double Profile::detInferProcessTime_ = 0;
double Profile::recInferProcessTime_ = 0;
double Profile::clsInferProcessTime_ = 0;

Profile &Profile::GetInstance() {
  std::unique_lock<std::mutex> lock(gMtx);
  static Profile singleton;
  return singleton;
}

std::atomic_int &Profile::GetStoppedThreadNum() {
  return stoppedThreadNum_;
}

int Profile::GetThreadNum() const {
  return threadNum_;
}

void Profile::SetThreadNum(int num) {
  threadNum_ = num;
}
