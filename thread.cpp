#include "thread.hpp"

ThreadGuard::ThreadGuard(std::thread& t) : t_{t} {}

ThreadGuard::~ThreadGuard() {
  if (t_.joinable()) {
    t_.join();
  }
}