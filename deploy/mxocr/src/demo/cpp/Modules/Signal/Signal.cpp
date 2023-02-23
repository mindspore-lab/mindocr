/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: signal control.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "Signal.h"

static std::mutex g_mtx = {};
bool Signal::signalRecieved = false;
double Signal::detInferTime = 0;
double Signal::recInferTime = 0;
double Signal::e2eProcessTime = 0;

double Signal::detPreProcessTime = 0;
double Signal::detPostProcessTime = 0;

double Signal::clsPreProcessTime = 0;
double Signal::clsInferTime = 0;
double Signal::clsPostProcessTime = 0;

double Signal::recPreProcessTime = 0;
double Signal::recPostProcessTime = 0;

double Signal::detInferProcessTime = 0;
double Signal::recInferProcessTime = 0;
double Signal::clsInferProcessTime = 0;

Signal &Signal::GetInstance()
{
    std::unique_lock<std::mutex> lock(g_mtx);
    static Signal singleton;
    return singleton;
}

std::atomic_int &Signal::GetStopedThreadNum()
{
    return stopedThreadNum;
}

int Signal::GetThreadNum() const
{
    return threadNum;
}

void Signal::SetThreadNum(int num)
{
    threadNum = num;
}
