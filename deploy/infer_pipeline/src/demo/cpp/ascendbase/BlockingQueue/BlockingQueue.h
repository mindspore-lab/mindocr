/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include "ErrorCode/ErrorCode.h"
#include <condition_variable>
#include <list>
#include <mutex>
#include <stdint.h>

static const int DEFAULT_MAX_QUEUE_SIZE = 256;

template<typename T> class BlockingQueue {
public:
    BlockingQueue(uint32_t maxSize = DEFAULT_MAX_QUEUE_SIZE) : max_size_(maxSize), is_stoped_(false) {}

    ~BlockingQueue() {}

    APP_ERROR Pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.empty() && !is_stoped_) {
            empty_cond_.wait(lock);
        }

        if (is_stoped_) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.empty()) {
            return APP_ERR_QUEUE_EMPTY;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        full_cond_.notify_one();

        return APP_ERR_OK;
    }

    APP_ERROR Pop(T& item, unsigned int timeOutMs)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto realTime = std::chrono::milliseconds(timeOutMs);

        while (queue_.empty() && !is_stoped_) {
            empty_cond_.wait_for(lock, realTime);
        }
        
        if (is_stoped_) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.empty()) {
            return APP_ERR_QUEUE_EMPTY;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        full_cond_.notify_one();

        return APP_ERR_OK;
    }

    APP_ERROR Push(const T& item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.size() >= max_size_ && isWait && !is_stoped_) {
            full_cond_.wait(lock);
        }

        if (is_stoped_) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.size() >= max_size_) {
            return APP_ERR_QUEUE_FULL;
        }
        queue_.push_back(item);

        empty_cond_.notify_one();

        return APP_ERR_OK;
    }

    APP_ERROR Push_Front(const T &item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.size() >= max_size_ && isWait && !is_stoped_) {
            full_cond_.wait(lock);
        }

        if (is_stoped_) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.size() >= max_size_) {
            return APP_ERR_QUEUE_FULL;
        }

        queue_.push_front(item);

        empty_cond_.notify_one();

        return APP_ERR_OK;
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_stoped_ = true;
        }

        full_cond_.notify_all();
        empty_cond_.notify_all();
    }

    void Restart()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_stoped_ = false;
        }
    }

    // if the queue is stoped ,need call this function to release the unprocessed items
    std::list<T> GetRemainItems()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!is_stoped_) {
            return std::list<T>();
        }

        return queue_;
    }

    APP_ERROR GetBackItem(T &item)
    {
        if (is_stoped_) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.empty()) {
            return APP_ERR_QUEUE_EMPTY;
        }

        item = queue_.back();
        return APP_ERR_OK;
    }

    const std::mutex *GetLock()
    {
        return &mutex_;
    }

    APP_ERROR IsFull()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size() >= max_size_;
    }

    int GetSize()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    APP_ERROR IsEmpty()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void Clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.clear();
    }

private:
    std::list<T> queue_;
    std::mutex mutex_;
    std::condition_variable empty_cond_;
    std::condition_variable full_cond_;
    uint32_t max_size_;

    bool is_stoped_;
};
#endif // __INC_BLOCKING_QUEUE_H__
