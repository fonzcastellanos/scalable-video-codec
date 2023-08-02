#ifndef SCALABLE_VIDEO_CODEC_THREAD_HPP
#define SCALABLE_VIDEO_CODEC_THREAD_HPP

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

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

namespace this_thread {
void InterruptionPoint();
};

class InterruptFlag {
 public:
  InterruptFlag();

  void Set();
  bool IsSet() const;
  void SetCondVar(std::condition_variable& cv);
  void ClearCondVar();

  struct ClearCondVarOnDestruct {
    ~ClearCondVarOnDestruct();
  };

  template <typename Lockable>
  void Wait(std::condition_variable_any& cv, Lockable& lk) {
    custom_lock cl{*this, cv, lk};
    this_thread::InterruptionPoint();
    cv.wait(cl);
    this_thread::InterruptionPoint();
  }

  template <typename Lockable, typename Predicate>
  void Wait(std::condition_variable_any& cv, Lockable& lk, Predicate pred) {
    custom_lock cl{*this, cv, lk};
    this_thread::InterruptionPoint();
    while (!IsSet() && !pred()) {
      cv.wait(cl);
    }
    this_thread::InterruptionPoint();
  }

 private:
  template <typename Lockable>
  class custom_lock {
   public:
    custom_lock(InterruptFlag& self, std::condition_variable_any& cv,
                Lockable& lk)
        : self_{self}, lk_{lk} {
      self_.thread_cond_mutex_.lock();
      self_.thread_cond_any_ = &cv;
    }
    ~custom_lock() {
      self_.thread_cond_any_ = 0;
      self_.thread_cond_mutex_.unlock();
    }
    void lock() { std::lock(self_.thread_cond_mutex_, lk_); }
    void unlock() {
      lk_.unlock();
      self_.thread_cond_mutex_.unlock();
    }

    InterruptFlag& self_;
    Lockable& lk_;
  };

  std::atomic<bool> flag_;
  std::condition_variable* thread_cond_;
  std::condition_variable_any* thread_cond_any_;
  std::mutex thread_cond_mutex_;
};

namespace this_thread {
inline thread_local InterruptFlag interrupt_flag;
};

template <typename Lockable>
void InterruptibleWait(std::condition_variable_any& cv, Lockable& lk) {
  this_thread::interrupt_flag.Wait(cv, lk);
}

template <typename Lockable, typename Predicate>
void InterruptibleWait(std::condition_variable_any& cv, Lockable& lk,
                       Predicate pred) {
  this_thread::interrupt_flag.Wait(cv, lk, pred);
}

class ThreadInterrupted {};

class IJThread {
 public:
  template <typename FunctionType>
  IJThread(FunctionType func)
      : flag_{std::make_unique<InterruptFlag*>()},
        flag_mutex_{std::make_unique<std::mutex>()} {
    std::promise<InterruptFlag*> p;
    auto future = p.get_future();
    thread_ = std::thread([func, &fl = *flag_, &fl_mutex = *flag_mutex_,
                           prom = std::move(p)]() mutable {
      prom.set_value(&this_thread::interrupt_flag);
      try {
        func();
      } catch (const ThreadInterrupted&) {
      }
      std::lock_guard g{fl_mutex};
      fl = 0;
    });
    *flag_ = future.get();
  }

  IJThread(const IJThread&) = delete;
  IJThread& operator=(const IJThread&) = delete;

  IJThread(IJThread&&);
  IJThread& operator=(IJThread&&);

  ~IJThread();

  void Interrupt();

 private:
  std::thread thread_;
  std::unique_ptr<InterruptFlag*> flag_;
  std::unique_ptr<std::mutex> flag_mutex_;
};

class InterruptGuard {
 public:
  InterruptGuard(std::vector<IJThread> threads);
  ~InterruptGuard();

 private:
  std::vector<IJThread> threads_;
};

#endif  // SCALABLE_VIDEO_CODEC_THREAD_HPP
