#ifndef SCALABLE_VIDEO_CODEC_DECODER_HPP
#define SCALABLE_VIDEO_CODEC_DECODER_HPP

#include <opencv2/core/mat.hpp>
#include <shared_mutex>
#include <vector>

#include "types.hpp"

const char* kWindowName = "Decoded Video";

struct DecoderConfig {
  uint foreground_quant_step;
  uint background_quant_step;
  uint max_gaze_rect_w;
  uint max_gaze_rect_h;
};

struct Block {
  uint type;
  std::vector<cv::Mat1f> channels;
};

struct SharedVec2 {
  int x;
  int y;
  std::shared_mutex mutex;
};

#endif  // SCALABLE_VIDEO_CODEC_DECODER_HPP