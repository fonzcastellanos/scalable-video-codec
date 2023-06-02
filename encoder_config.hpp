#ifndef SCALABLE_VIDEO_CODEC_ENCODER_CONFIG_HPP
#define SCALABLE_VIDEO_CODEC_ENCODER_CONFIG_HPP

#include "codec.hpp"
#include "motion.hpp"
#include "types.hpp"

struct KMeansParams {
  uint cluster_count;
  uint attempt_count;
  uint max_iter_count;
  float epsilon;
};

struct EncoderConfig {
  uint mv_block_w;
  uint mv_block_h;
  uint mv_search_range;
  uint pyr_lvl_count;
  RansacParams ransac;
  uint morph_rect_w;
  uint morph_rect_h;
  KMeansParams kmeans;
  uint connected_components_connectivity;
  uint transform_block_w;
  uint transform_block_h;
  uint frame_w;
  uint frame_h;
};

struct Config {
  char* video_path;
  boolean verbose;
  EncoderConfig encoder;
};

Status ParseConfig(uint argc, char* argv[], Config* c);

void DefaultInit(KMeansParams* p);
void DefaultInit(RansacParams* p);
void DefaultInit(EncoderConfig* c);
void DefaultInit(Config* c);

Status Validate(KMeansParams* p);
Status Validate(EncoderConfig* c);
Status Validate(EncoderConfig* c);

#endif  // SCALABLE_VIDEO_CODEC_ENCODER_CONFIG_HPP