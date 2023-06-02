#ifndef SCALABLE_VIDEO_CODEC_DECODER_CONFIG_HPP
#define SCALABLE_VIDEO_CODEC_DECODER_CONFIG_HPP

#include "codec.hpp"
#include "types.hpp"

struct Config {
  uint foreground_quant_step;
  uint background_quant_step;
  uint max_gaze_rect_w;
  uint max_gaze_rect_h;
};

Status ParseConfig(int argc, char* argv[], Config* c);

void DefaultInit(Config* c);

Status Validate(Config* c);

#endif  // SCALABLE_VIDEO_CODEC_DECODER_CONFIG_HPP
