#ifndef SCALABLE_VIDEO_CODEC_DRAW_HPP
#define SCALABLE_VIDEO_CODEC_DRAW_HPP

#include <opencv2/core.hpp>

#include "math.hpp"
#include "types.hpp"

struct ArrowedLineParams {
  cv::Scalar color;
  uint thickness;
  uint shift;
  double tip_len;
  cv::LineTypes line_type;
};

void DefaultInit(ArrowedLineParams* p);

void DrawMotionField(const Vec2f* motion_field, uint block_w, uint block_h,
                     const ArrowedLineParams* line_params, cv::Mat* frame);

void DrawMotionVecAsField(Vec2f motion_vec, uint block_w, uint block_h,
                          const ArrowedLineParams* line_params, cv::Mat* frame);

void DrawVecFieldLayerClusters(const uint* cluster_ids, uint min_cluster_id,
                               const uint* field_indices, uint field_indices_sz,
                               uint field_w, uint field_h, uint block_w,
                               uint block_h, cv::Mat3b* frame);

void DrawOutlinedText(cv::Mat* frame, const char* text, cv::Point origin,
                      cv::Scalar outline_color, cv::Scalar fill_color,
                      cv::HersheyFonts font, float font_scale_factor,
                      float line_thickness_scale_factor,
                      cv::LineTypes line_type);

#endif  // SCALABLE_VIDEO_CODEC_DRAW_HPP