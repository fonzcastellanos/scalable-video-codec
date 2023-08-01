#ifndef SCALABLE_VIDEO_CODEC_QUEUE_HPP
#define SCALABLE_VIDEO_CODEC_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

#include "thread.hpp"

template <typename T>
class CircularQueue {
 public:
  CircularQueue(std::size_t capacity)
      : buffer_{std::make_unique<T[]>(capacity)},
        capacity_{capacity},
        size_{},
        read_idx_{},
        write_idx_{},
        is_producer_done_{} {}

  void Push(T item) {
    std::unique_lock<std::mutex> lk{mutex_};

    // InterruptibleWait(is_not_full_, lock,
    //                   [this]() { return size_ != capacity_; });

    is_not_full_.wait(lk, [this]() { return size_ != capacity_; });

    buffer_[write_idx_] = std::move(item);
    write_idx_ = (write_idx_ + 1) % capacity_;
    ++size_;

    is_not_empty_or_is_producer_done_.notify_one();
  }

  bool Pop(T& item) {
    std::unique_lock<std::mutex> lk{mutex_};

    // InterruptibleWait(is_not_empty_, lock, [this]() { return size_ != 0; });

    is_not_empty_or_is_producer_done_.wait(
        lk, [this]() { return size_ != 0 || is_producer_done_; });
    if (is_producer_done_) {
      return false;
    }

    item = std::move(buffer_[read_idx_]);
    read_idx_ = (read_idx_ + 1) % capacity_;
    --size_;

    is_not_full_.notify_one();

    return true;
  }

  void SignalProducerIsDone() {
    std::lock_guard<std::mutex> l{mutex_};
    is_producer_done_ = true;
    is_not_empty_or_is_producer_done_.notify_one();
  }

  bool IsEmpty() {
    std::lock_guard<std::mutex> l{mutex_};
    return size_ == 0;
  }

  bool IsFull() {
    std::lock_guard<std::mutex> l{mutex_};
    return size_ == capacity_;
  }

  std::size_t Size() {
    std::lock_guard<std::mutex> l{mutex_};
    return size_;
  }

 private:
  std::unique_ptr<T[]> buffer_;
  std::size_t capacity_;
  std::size_t size_;
  std::size_t read_idx_;
  std::size_t write_idx_;
  bool is_producer_done_;
  std::mutex mutex_;
  std::condition_variable is_not_full_;
  std::condition_variable is_not_empty_or_is_producer_done_;
  // std::condition_variable_any is_not_full_;
  // std::condition_variable_any is_not_empty_;
};

#endif  // SCALABLE_VIDEO_CODEC_QUEUE_HPP