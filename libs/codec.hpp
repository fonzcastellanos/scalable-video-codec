#ifndef SCALABLE_VIDEO_CODEC_CODEC_HPP
#define SCALABLE_VIDEO_CODEC_CODEC_HPP

#include "types.hpp"

#define BLOCK_TYPE_BACKGROUND 0

struct Header {
  uint frame_count;
  uint frame_w;
  uint frame_h;
  uint frame_excess_w;
  uint frame_excess_h;
  uint transform_block_w;
  uint transform_block_h;
  uint channel_count;
};

#endif  // SCALABLE_VIDEO_CODEC_CODEC_HPP