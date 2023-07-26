#include "thread.hpp"

ThreadGuard::ThreadGuard(std::thread& t) : t_{t} {}

ThreadGuard::~ThreadGuard() {
  if (t_.joinable()) {
    t_.join();
  }
}

InterruptFlag::InterruptFlag()
    : flag_{0}, thread_cond_{}, thread_cond_any_{} {};

void InterruptFlag::Set() {
  flag_.store(true, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk{thread_cond_mutex_};
  if (thread_cond_) {
    thread_cond_->notify_all();
  } else if (thread_cond_any_) {
    thread_cond_any_->notify_all();
  }
}

bool InterruptFlag::IsSet() const {
  return flag_.load(std::memory_order_relaxed);
}

void InterruptFlag::SetCondVar(std::condition_variable& cv) {
  std::lock_guard<std::mutex> lk{thread_cond_mutex_};
  thread_cond_ = &cv;
}

void InterruptFlag::ClearCondVar() {
  std::lock_guard<std::mutex> lk{thread_cond_mutex_};
  thread_cond_ = 0;
}

InterruptFlag::ClearCondVarOnDestruct::~ClearCondVarOnDestruct() {
  this_thread::interrupt_flag.ClearCondVar();
}

void this_thread::InterruptionPoint() {
  if (this_thread::interrupt_flag.IsSet()) {
    throw ThreadInterrupted{};
  }
}

IJThread::IJThread(IJThread&& t)
    : thread_{std::move(t.thread_)},
      flag_{std::move(t.flag_)},
      flag_mutex_{std::move(t.flag_mutex_)} {}

IJThread& IJThread::operator=(IJThread&& t) {
  if (this == &t) {
    return *this;
  }

  if (thread_.joinable()) {
    thread_.join();
  }

  thread_ = std::move(t.thread_);
  flag_ = std::move(t.flag_);
  flag_mutex_ = std::move(t.flag_mutex_);

  return *this;
}

IJThread::~IJThread() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

void IJThread::Interrupt() {
  if (*flag_) {
    (*flag_)->Set();
  }
}

InterruptGuard::InterruptGuard(std::vector<IJThread> threads)
    : threads_{std::move(threads)} {}

InterruptGuard::~InterruptGuard() {
  for (auto& t : threads_) {
    t.Interrupt();
  }
}
