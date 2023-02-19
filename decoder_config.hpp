#ifndef SCALABLE_VIDEO_CODEC_DECODER_CONFIG_HPP
#define SCALABLE_VIDEO_CODEC_DECODER_CONFIG_HPP

#include <cassert>
#include <cstdio>

#include "codec.hpp"
#include "types.hpp"

struct Config {
  uint foreground_quant_step;
  uint background_quant_step;
  uint max_gaze_rect_w;
  uint max_gaze_rect_h;
};

CodecStatus ParseConfig(int argc, char* argv[], Config* c);

void DefaultInit(Config* c);

CodecStatus Validate(Config* c);

#endif  // SCALABLE_VIDEO_CODEC_DECODER_CONFIG_HPP
