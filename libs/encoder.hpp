#ifndef SCALABLE_VIDEO_CODEC_ENCODER_HPP
#define SCALABLE_VIDEO_CODEC_ENCODER_HPP

#include <future>
#include <opencv2/core/mat.hpp>
#include <vector>

#include "error.hpp"
#include "math.hpp"
#include "motion.hpp"
#include "queue.hpp"
#include "types.hpp"

Error Validate(const RansacParams&);

struct KMeansParams {
  uint cluster_count;
  uint attempt_count;
  uint max_iter_count;
  float epsilon;
};

Error Validate(const KMeansParams&);

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
};

Error Validate(const EncoderConfig&);

struct EncodedFrame {
  std::vector<cv::Mat1f> dct_coeffs;
  std::vector<uint> mv_field_block_types;
};

struct VideoProperties {
  uint frame_w;
  uint frame_h;
  uint frame_count;
};

class Encoder {
 public:
  Encoder(const EncoderConfig& cfg, const VideoProperties& vidprops,
          CircularQueue<cv::Mat3b>& in_queue,
          std::future<void> attempted_first_frame_read,
          CircularQueue<std::vector<uchar>>& out_queue);
  void operator()();

 private:
  EncoderConfig cfg_;
  VideoProperties vidprops_;
  CircularQueue<cv::Mat3b>& in_queue_;
  std::future<void> attempted_first_frame_read_;
  CircularQueue<std::vector<uchar>>& out_queue_;

  uint padded_frame_w_;
  uint padded_frame_h_;
  uint frame_excess_w_;
  uint frame_excess_h_;

  uint mv_field_w_;
  uint mv_field_h_;

  std::vector<Vec2f> mv_field_;
  std::vector<float> mv_field_min_mad_;

  cv::Mat1b foreground_mv_field_mask_;
  std::vector<uint> foreground_mv_field_indices_;
  std::vector<Vec4f> foreground_mv_features_;
  std::vector<uint> mv_field_block_types_;

  cv::Mat1b foreground_cluster_mask_;
  cv::Mat morph_rect_;

  cv::Mat3b padded_frame_;
  cv::Mat3b yuv_padded_frame_;
  cv::Mat1b prev_y_padded_frame_;
  cv::Mat1b y_padded_frame_;

  std::vector<cv::Mat1b> prev_pyr_;
  std::vector<uchar*> prev_pyr_data_;
  std::vector<cv::Mat1b> pyr_;
  std::vector<uchar*> pyr_data_;
};

#endif  // SCALABLE_VIDEO_CODEC_ENCODER_HPP