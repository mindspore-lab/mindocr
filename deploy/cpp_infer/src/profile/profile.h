#ifndef DEPLOY_CPP_INFER_SRC_PROFILE_PROFILE_H_
#define DEPLOY_CPP_INFER_SRC_PROFILE_PROFILE_H_

#include <atomic>
#include <mutex>

class Profile {
 public:
    Profile(const Profile &) = delete;

    Profile operator = (const Profile &) = delete;

    ~Profile() = default;

    static Profile &GetInstance();

    int GetThreadNum() const;

    void SetThreadNum(int num);

    std::atomic_int &GetStoppedThreadNum();

    static bool signalReceived_;
    static double detPreProcessTime_;
    static double detInferTime_;
    static double detPostProcessTime_;

    static double clsPreProcessTime_;
    static double clsInferTime_;
    static double clsPostProcessTime_;

    static double recPreProcessTime_;
    static double recInferTime_;
    static double recPostProcessTime_;
    static double e2eProcessTime_;

    static double detInferProcessTime_;
    static double recInferProcessTime_;
    static double clsInferProcessTime_;

 private:
    std::atomic_int threadNum_ {0 };
    std::atomic_int stoppedThreadNum_ {0 };

    Profile() {}
};

#endif
