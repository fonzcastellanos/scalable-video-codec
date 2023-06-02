#ifndef SCALABLE_VIDEO_CODEC_ENCODER_HPP
#define SCALABLE_VIDEO_CODEC_ENCODER_HPP

#include <opencv2/core/mat.hpp>
#include <vector>

#include "codec.hpp"
#include "math.hpp"
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

struct Encoder {
  EncoderConfig cfg;

  uint padded_frame_w;
  uint padded_frame_h;

  uint mv_field_w;
  uint mv_field_h;

  std::vector<Vec2f> mv_field;
  std::vector<float> mv_field_min_mad;

  std::vector<uchar> foreground_mv_field_mask;
  std::vector<uint> foreground_mv_field_indices;
  std::vector<Vec4f> foreground_mv_features;
  std::vector<uint> mv_field_block_types;

  cv::Mat3b padded_frame;
  cv::Mat3b yuv_padded_frame;
  cv::Mat1b prev_y_padded_frame;
  cv::Mat1b y_padded_frame;

  std::vector<cv::Mat1b> prev_pyr;
  std::vector<uchar*> prev_pyr_data;
  std::vector<cv::Mat1b> pyr;
  std::vector<uchar*> pyr_data;
};

void DefaultInit(KMeansParams* p);
void DefaultInit(RansacParams* p);
void DefaultInit(EncoderConfig* c);

Status Validate(KMeansParams* p);
Status Validate(RansacParams* c);
Status Validate(EncoderConfig* c);

void Init(Encoder* e, EncoderConfig* cfg);

#endif  // SCALABLE_VIDEO_CODEC_ENCODER_HPP