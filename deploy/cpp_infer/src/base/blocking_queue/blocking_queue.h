#ifndef DEPLOY_CPP_INFER_SRC_BASE_BLOCKING_QUEUE_BLOCKING_QUEUE_H_
#define DEPLOY_CPP_INFER_SRC_BASE_BLOCKING_QUEUE_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <list>
#include <mutex>
#include <cstdint>
#include "status_code/status_code.h"

static const int DEFAULT_MAX_SIZE = 256;

template<typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(uint32_t maxSize = DEFAULT_MAX_SIZE) : maxSize_(maxSize), isStopped_(false) {}

  ~BlockingQueue() = default;

  Status Pop(T *item) {
    std::unique_lock<std::mutex> lock(mutex_);

    while (queue_.empty() && !isStopped_) {
      emptyCond_.wait(lock);
    }

    if (isStopped_) {
      return Status::QUEUE_STOPPED;
    }

    if (queue_.empty()) {
      return Status::QUEUE_EMPTY;
    } else {
      *item = queue_.front();
      queue_.pop_front();
    }

    fullCond_.notify_one();

    return Status::OK;
  }

  Status Pop(T *item, unsigned int timeOutMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto realTime = std::chrono::milliseconds(timeOutMs);

    while (queue_.empty() && !isStopped_) {
      emptyCond_.wait_for(lock, realTime);
    }

    if (isStopped_) {
      return Status::QUEUE_STOPPED;
    }

    if (queue_.empty()) {
      return Status::QUEUE_EMPTY;
    } else {
      *item = queue_.front();
      queue_.pop_front();
    }

    fullCond_.notify_one();

    return Status::OK;
  }

  Status Push(const T &item, bool isWait = false) {
    std::unique_lock<std::mutex> lock(mutex_);

    while (queue_.size() >= maxSize_ && isWait && !isStopped_) {
      fullCond_.wait(lock);
    }

    if (isStopped_) {
      return Status::QUEUE_STOPPED;
    }

    if (queue_.size() >= maxSize_) {
      return Status::QUEUE_FULL;
    }
    queue_.push_back(item);

    emptyCond_.notify_one();

    return Status::OK;
  }

  Status PushFront(const T &item, bool isWait = false) {
    std::unique_lock<std::mutex> lock(mutex_);

    while (queue_.size() >= maxSize_ && isWait && !isStopped_) {
      fullCond_.wait(lock);
    }

    if (isStopped_) {
      return Status::QUEUE_STOPPED;
    }

    if (queue_.size() >= maxSize_) {
      return Status::QUEUE_FULL;
    }

    queue_.push_front(item);

    emptyCond_.notify_one();

    return Status::OK;
  }

  void Stop() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      isStopped_ = true;
    }

    fullCond_.notify_all();
    emptyCond_.notify_all();
  }

  void Restart() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      isStopped_ = false;
    }
  }

  std::list<T> GetRemainItems() {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!isStopped_) {
      return std::list<T>();
    }

    return queue_;
  }

  Status GetBackItem(const T &item) {
    if (isStopped_) {
      return Status::QUEUE_STOPPED;
    }

    if (queue_.empty()) {
      return Status::QUEUE_EMPTY;
    }

    item = queue_.back();
    return Status::OK;
  }

  const std::mutex *GetLock() {
    return &mutex_;
  }

  Status IsFull() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size() >= maxSize_;
  }

  int GetSize() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

  Status IsEmpty() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void Clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.clear();
  }

 private:
  std::list<T> queue_;
  std::mutex mutex_;
  std::condition_variable emptyCond_;
  std::condition_variable fullCond_;
  uint32_t maxSize_;
  bool isStopped_;
};
#endif
