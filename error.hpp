#ifndef SCALABLE_VIDEO_CODEC_ERROR_HPP
#define SCALABLE_VIDEO_CODEC_ERROR_HPP

#include <string>
#include <vector>

enum class ErrorCode { kOk, kUnspecified, kInvalidParameter };

struct Error {
  ErrorCode code;
  std::string message;
};

#endif  // SCALABLE_VIDEO_CODEC_ERROR_HPP