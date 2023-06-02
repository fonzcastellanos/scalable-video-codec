#include "encoder.hpp"

#include <cassert>
#include <cstdio>

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

void DefaultInit(KMeansParams* p) {
  assert(p);

  p->cluster_count = 10;
  p->attempt_count = 3;
  p->max_iter_count = 10;
  p->epsilon = 1;
}

void DefaultInit(RansacParams* p) {
  assert(p);

  p->subset_sz = 1;
  p->inlier_ratio = 0.5f;
  p->success_prob = 0.99f;
  p->inlier_thresh = 7.5f;
}

void DefaultInit(EncoderConfig* c) {
  assert(c);

  c->mv_block_w = 16;
  c->mv_block_h = 16;
  c->mv_search_range = 8;
  c->pyr_lvl_count = 4;

  DefaultInit(&c->ransac);

  c->morph_rect_w = 3;
  c->morph_rect_h = 3;

  DefaultInit(&c->kmeans);

  c->connected_components_connectivity = 4;
  c->transform_block_w = 8;
  c->transform_block_h = 8;
}

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

Status Validate(KMeansParams* p) {
  assert(p);

  Status res = kStatus_InvalidParamError;

  if (p->cluster_count == 0) {
    std::fprintf(stderr, "number of clusters must be > 0\n");
    return res;
  }

  if (p->attempt_count == 0) {
    std::fprintf(stderr, "number of attempts must be > 0\n");
    return res;
  }

  if (p->max_iter_count == 0) {
    std::fprintf(stderr, "maximum number of iterations > 0\n");
    return res;
  }

  if (p->epsilon <= 0) {
    std::fprintf(stderr, "epsilon must be > 0\n");
    return res;
  }

  res = kStatus_Ok;

  return res;
}

Status Validate(RansacParams* p) {
  assert(p);

  Status res = kStatus_InvalidParamError;

  if (p->inlier_thresh < 0) {
    std::fprintf(stderr, "inlier threshold must be >= 0");
    return res;
  }

  if (p->success_prob < 0) {
    std::fprintf(stderr, "success probability must be >= 0");
    return res;
  }

  if (p->inlier_ratio < 0) {
    std::fprintf(stderr, "inlier ratio must be >= 0");
    return res;
  }

  res = kStatus_Ok;

  return res;
}

Status Validate(EncoderConfig* c) {
  assert(c);

  Status status = kStatus_InvalidParamError;

  if (c->mv_block_w < 1) {
    std::fprintf(stderr, "MV block width must be > 0\n");
    return status;
  }

  if (c->mv_block_h < 1) {
    std::fprintf(stderr, "MV block height must be > 0\n");
    return status;
  }

  if (c->pyr_lvl_count < 1) {
    std::fprintf(stderr, "number of pyramid levels must be > 0\n");
    return status;
  }

  uint top_lvl_reduction_factor = Pow2(c->pyr_lvl_count - 1);
  if (c->mv_search_range / top_lvl_reduction_factor == 0) {
    std::fprintf(stderr,
                 "the quotient from dividing the MV search range by the "
                 "top pyramid level reduction factor must be > 0\n");
    return status;
  }

  status = Validate(&c->ransac);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to validate RANSAC parameters\n");
    return status;
  }

  status = Validate(&c->kmeans);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to validate k-means parameters\n");
    return status;
  }

  status = kStatus_InvalidParamError;

  if (c->connected_components_connectivity != 4 &&
      c->connected_components_connectivity != 8) {
    std::fprintf(stderr,
                 "connected components connectivity must be either 4 or 8\n");
    return status;
  }

  if (c->transform_block_w < 1) {
    std::fprintf(stderr, "transform block width must be > 0\n");
    return status;
  }

  if (c->transform_block_h < 1) {
    std::fprintf(stderr, "transform block height must be > 0\n");
    return status;
  }

  // IF the width and height of transform blocks are greater than the width
  // and height, respectively, of motion blocks, AND the width and height of
  // motion blocks are not divisible by the width and height, respectively, of
  // transform blocks, THEN the mapping of block types from motion blocks to
  // transform blocks would be ambiguous because a transform block would
  // overlap with multiple motion blocks.
  if (c->transform_block_w > c->mv_block_w) {
    std::fprintf(stderr,
                 "transform block width must be <= motion block width\n");
    return status;
  }
  if (c->transform_block_h > c->mv_block_h) {
    std::fprintf(stderr,
                 "transform block height must be <= motion block height\n");
    return status;
  }
  if (c->mv_block_w % c->transform_block_w != 0) {
    std::fprintf(
        stderr,
        "motion block width must be divisible by transform block width\n");
    return status;
  }
  if (c->mv_block_h % c->transform_block_h != 0) {
    std::fprintf(
        stderr,
        "motion block height must be divisible by transform block height\n");
    return status;
  }

  status = kStatus_Ok;

  return status;
}

void Init(Encoder* e, EncoderConfig* cfg) {
  assert(e);
  assert(cfg);

  e->cfg = *cfg;

  // The notation a|b denotes a divides b.
  // Theorem: If a|b and b|c, then a|c.
  //
  // Therefore, if the MV block width and height are divisible by the
  // transform block width and height, respectively, and the frame width
  // and height are made divisible by the MV block width and height,
  // respectively, then the frame width and height are also divisible by the
  // transform block width and height. That's why there's no need to
  // involve the transform block dimensions in these calculations. So
  // long as the MV block width and height are divisible by the
  // transform block width and height, we're good. Validation of
  // configuration ensures this.
  uint top_lvl_reduction_factor = Pow2(cfg->pyr_lvl_count - 1);
  e->padded_frame_w = ClosestLargerDivisible(cfg->frame_w, cfg->mv_block_w,
                                             top_lvl_reduction_factor);
  e->padded_frame_h = ClosestLargerDivisible(cfg->frame_h, cfg->mv_block_h,
                                             top_lvl_reduction_factor);

  e->mv_field_w = e->padded_frame_w / cfg->mv_block_w;
  e->mv_field_h = e->padded_frame_h / cfg->mv_block_h;

  uint mv_field_sz = e->mv_field_w * e->mv_field_h;

  e->mv_field.resize(mv_field_sz);
  e->mv_field_min_mad.resize(mv_field_sz);

  e->foreground_mv_field_mask.resize(mv_field_sz);
  e->foreground_mv_field_indices.reserve(mv_field_sz);
  e->foreground_mv_features.reserve(mv_field_sz);

  e->mv_field_block_types.resize(mv_field_sz);

  e->padded_frame = cv::Mat3b(e->padded_frame_h, e->padded_frame_w);
  e->yuv_padded_frame = cv::Mat3b(e->padded_frame_h, e->padded_frame_w);
  e->prev_y_padded_frame = cv::Mat1b(e->padded_frame_h, e->padded_frame_w);
  e->y_padded_frame = cv::Mat1b(e->padded_frame_h, e->padded_frame_w);

  e->prev_pyr.resize(cfg->pyr_lvl_count);
  e->prev_pyr_data.resize(cfg->pyr_lvl_count);
  e->pyr.resize(cfg->pyr_lvl_count);
  e->pyr_data.resize(cfg->pyr_lvl_count);

  e->prev_pyr[0] = e->prev_y_padded_frame;
  e->prev_pyr_data[0] = e->prev_pyr[0].data;
  e->pyr[0] = e->y_padded_frame;
  e->pyr_data[0] = e->pyr[0].data;
  {
    uint w = e->padded_frame_w / 2;
    uint h = e->padded_frame_h / 2;
    for (uint i = 1; i < cfg->pyr_lvl_count; ++i) {
      e->prev_pyr[i] = cv::Mat1b(h, w);
      e->prev_pyr_data[i] = e->prev_pyr[i].data;

      e->pyr[i] = cv::Mat1b(h, w);
      e->pyr_data[i] = e->pyr[i].data;

      w /= 2;
      h /= 2;
    }
  }
}
