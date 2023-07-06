#ifndef SCALABLE_VIDEO_CODEC_DECODER_HPP
#define SCALABLE_VIDEO_CODEC_DECODER_HPP

#include "codec.hpp"
#include "types.hpp"

struct DecoderConfig {
  uint foreground_quant_step;
  uint background_quant_step;
  uint max_gaze_rect_w;
  uint max_gaze_rect_h;
};

void DefaultInit(DecoderConfig*);

Status Validate(DecoderConfig*);

#endif  // SCALABLE_VIDEO_CODEC_DECODER_HPP