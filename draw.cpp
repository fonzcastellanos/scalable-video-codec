#include "draw.hpp"

#include <cassert>
#include <opencv2/imgproc.hpp>

void DefaultInit(ArrowedLineParams* p) {
  assert(p);

  p->color = cv::Scalar(20, 255, 57);
  p->thickness = 1;
  p->shift = 0;
  p->tip_len = 0.2;
  p->line_type = cv::LineTypes::LINE_AA;
}

/* 12 Visually distinct colors */
// const cv::Scalar kColorTab[] = {
//     cv::Scalar(79, 79, 47),    cv::Scalar(19, 69, 139),
//     cv::Scalar(34, 139, 34),   cv::Scalar(139, 0, 0),
//     cv::Scalar(0, 0, 255),     cv::Scalar(0, 255, 255),
//     cv::Scalar(0, 255, 0),     cv::Scalar(255, 255, 0),
//     cv::Scalar(255, 0, 255),   cv::Scalar(237, 149, 100),
//     cv::Scalar(181, 228, 255), cv::Scalar(180, 105, 255)};

/* The hexadecimal version of kColorPalette */
// const char* const kHexColorPalette[] = {
//     "a9a9a9", "2f4f4f", "556b2f", "228b22", "800000", "808000",
//     "483d8b", "008b8b", "000080", "9acd32", "7f007f", "8fbc8f",
//     "b03060", "ff4500", "ffa500", "ffff00", "7fff00", "9400d3",
//     "00ff7f", "dc143c", "00ffff", "00bfff", "f4a460", "0000ff",
//     "ff00ff", "f0e68c", "fa8072", "6495ed", "dda0dd", "90ee90",
//     "ff1493", "7b68ee", "afeeee", "ee82ee", "ffe4c4", "ffb6c1"};

/* 36 Visually distinct color palette */
const cv::Scalar kColorPalette[] = {
    cv::Scalar(169, 169, 169), cv::Scalar(79, 79, 47),
    cv::Scalar(47, 107, 85),   cv::Scalar(34, 139, 34),
    cv::Scalar(0, 0, 128),     cv::Scalar(0, 128, 128),
    cv::Scalar(139, 61, 72),   cv::Scalar(139, 139, 0),
    cv::Scalar(128, 0, 0),     cv::Scalar(50, 205, 154),
    cv::Scalar(127, 0, 127),   cv::Scalar(143, 188, 143),
    cv::Scalar(96, 48, 176),   cv::Scalar(0, 69, 255),
    cv::Scalar(0, 165, 255),   cv::Scalar(0, 255, 255),
    cv::Scalar(0, 255, 127),   cv::Scalar(211, 0, 148),
    cv::Scalar(127, 255, 0),   cv::Scalar(60, 20, 220),
    cv::Scalar(255, 255, 0),   cv::Scalar(255, 191, 0),
    cv::Scalar(96, 164, 244),  cv::Scalar(255, 0, 0),
    cv::Scalar(255, 0, 255),   cv::Scalar(140, 230, 240),
    cv::Scalar(114, 128, 250), cv::Scalar(237, 149, 100),
    cv::Scalar(221, 160, 221), cv::Scalar(144, 238, 144),
    cv::Scalar(147, 20, 255),  cv::Scalar(238, 104, 123),
    cv::Scalar(238, 238, 175), cv::Scalar(238, 130, 238),
    cv::Scalar(196, 228, 255), cv::Scalar(193, 182, 255)};

void DrawMotionField(const Vec2f* motion_field, uint block_w, uint block_h,
                     const ArrowedLineParams* line_params, cv::Mat* frame) {
  assert(motion_field);
  assert(line_params);
  assert(frame);

  assert(block_w != 0);
  assert(block_h != 0);

  assert(frame->cols % block_w == 0);
  assert(frame->rows % block_h == 0);

  uint motion_field_w = frame->cols / block_w;
  uint motion_field_h = frame->rows / block_h;

  for (uint mf_y = 0; mf_y < motion_field_h; ++mf_y) {
    uint y = mf_y * block_h;

    for (uint mf_x = 0; mf_x < motion_field_w; ++mf_x) {
      uint x = mf_x * block_w;

      uint mf_i = mf_y * motion_field_w + mf_x;

      Vec2i mv = Vec2fToVec2i(motion_field[mf_i]);
      cv::Vec2i cv_mv(mv.x, mv.y);

      cv::Vec2i from(x, y);
      cv::Vec2i to = from + cv_mv;

      cv::arrowedLine(*frame, from, to, line_params->color,
                      line_params->thickness, line_params->line_type,
                      line_params->shift, line_params->tip_len);
    }
  }
}

void DrawMotionVecAsField(Vec2f motion, uint block_w, uint block_h,
                          const ArrowedLineParams* line_params,
                          cv::Mat* frame) {
  assert(frame);
  assert(line_params);

  assert(block_w != 0);
  assert(block_h != 0);

  assert(frame->cols % block_w == 0);
  assert(frame->rows % block_h == 0);

  Vec2i mv = Vec2fToVec2i(motion);
  cv::Vec2i cv_mv(mv.x, mv.y);

  for (uint y = 0; y < frame->rows; y += block_h) {
    for (uint x = 0; x < frame->cols; x += block_w) {
      cv::Vec2i from(x, y);
      cv::Vec2i to = from + cv_mv;

      cv::arrowedLine(*frame, from, to, line_params->color,
                      line_params->thickness, line_params->line_type,
                      line_params->shift, line_params->tip_len);
    }
  }
}

void DrawVecFieldLayerClusters(const uint* cluster_ids, uint min_cluster_id,
                               const uint* field_indices, uint field_indices_sz,
                               uint field_w, uint field_h, uint block_w,
                               uint block_h, cv::Mat3b* frame) {
  assert(cluster_ids);
  assert(field_indices);
  assert(frame);

  for (uint i = 0; i < field_indices_sz; ++i) {
    uint fi = field_indices[i];

    uint fx = fi % field_w;
    uint fy = fi / field_w;

    int x = fx * block_w;
    int y = fy * block_h;

    uint color_pal_id = (cluster_ids[fi] - min_cluster_id) %
                        (sizeof(kColorPalette) / sizeof(cv::Scalar));

    cv::rectangle(*frame, cv::Rect(x, y, block_w, block_h),
                  kColorPalette[color_pal_id], cv::FILLED);
  }
}

void DrawOutlinedText(cv::Mat* frame, const char* text, cv::Point origin,
                      cv::Scalar outline_color, cv::Scalar fill_color,
                      cv::HersheyFonts font, float font_scale_factor,
                      float line_thickness_scale_factor,
                      cv::LineTypes line_type) {
  assert(frame);
  assert(text);

  uint outer_thickness = RoundFloatToInt(2 * line_thickness_scale_factor);
  uint inner_thickness = RoundFloatToInt(line_thickness_scale_factor);

  cv::putText(*frame, text, origin, font, font_scale_factor, outline_color,
              outer_thickness, line_type);
  cv::putText(*frame, text, origin, font, font_scale_factor, fill_color,
              inner_thickness, line_type);
}