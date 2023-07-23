#ifndef SCALABLE_VIDEO_CODEC_QUEUE_HPP
#define SCALABLE_VIDEO_CODEC_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

template <typename T>
class CircularQueue {
 public:
  using SizeType = std::size_t;

  CircularQueue(SizeType capacity)
      : buffer_{std::make_unique<T[]>(capacity)},
        capacity_{capacity},
        size_{},
        read_idx_{},
        write_idx_{} {}

  void Push(T item) {
    std::unique_lock<std::mutex> lock{mutex_};

    not_full_.wait(lock, [this]() { return size_ != capacity_; });

    buffer_[write_idx_] = std::move(item);
    write_idx_ = (write_idx_ + 1) % capacity_;
    ++size_;

    not_empty_.notify_one();
  }

  T Pop() {
    std::unique_lock<std::mutex> lock{mutex_};

    not_empty_.wait(lock, [this]() { return size_ != 0; });

    T item = std::move(buffer_[read_idx_]);
    read_idx_ = (read_idx_ + 1) % capacity_;
    --size_;

    not_full_.notify_one();

    return item;
  }

  bool IsEmpty() {
    std::lock_guard<std::mutex> lock{mutex_};
    return size_ == 0;
  }

  bool IsFull() {
    std::lock_guard<std::mutex> lock{mutex_};
    return size_ == capacity_;
  }

  SizeType Size() {
    std::lock_guard<std::mutex> lock{mutex_};
    return size_;
  }

 private:
  std::unique_ptr<T[]> buffer_;
  SizeType capacity_;
  SizeType size_;
  SizeType read_idx_;
  SizeType write_idx_;
  std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
};

#endif  // SCALABLE_VIDEO_CODEC_QUEUE_HPP