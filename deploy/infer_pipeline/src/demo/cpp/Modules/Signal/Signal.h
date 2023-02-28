/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: signal control.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef CPP_SIGNAL_H
#define CPP_SIGNAL_H

#include <atomic>
#include <mutex>

class Signal {
public:
    Signal(const Signal &) = delete;

    Signal operator = (const Signal &) = delete;

    ~Signal() {}

    static Signal &GetInstance();

    int GetThreadNum() const;

    void SetThreadNum(int num);

    std::atomic_int &GetStopedThreadNum();

    static bool signalRecieved;
    static double detPreProcessTime;
    static double detInferTime;
    static double detPostProcessTime;

    static double clsPreProcessTime;
    static double clsInferTime;
    static double clsPostProcessTime;

    static double recPreProcessTime;
    static double recInferTime;
    static double recPostProcessTime;
    static double e2eProcessTime;

    static double detInferProcessTime;
    static double recInferProcessTime;
    static double clsInferProcessTime;

private:
    std::atomic_int threadNum { 0 };
    std::atomic_int stopedThreadNum { 0 };

    Signal() {}
};

#endif
