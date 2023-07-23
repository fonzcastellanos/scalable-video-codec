#ifndef SCALABLE_VIDEO_CODEC_THREAD_HPP
#define SCALABLE_VIDEO_CODEC_THREAD_HPP

#include <thread>

class ThreadGuard {
 public:
  explicit ThreadGuard(std::thread& t);
  ~ThreadGuard();
  ThreadGuard(const ThreadGuard&) = delete;
  ThreadGuard& operator=(const ThreadGuard&) = delete;
  ThreadGuard(ThreadGuard&&) = delete;
  ThreadGuard& operator=(ThreadGuard&&) = delete;

 private:
  std::thread& t_;
};

#endif  // SCALABLE_VIDEO_CODEC_THREAD_HPP
