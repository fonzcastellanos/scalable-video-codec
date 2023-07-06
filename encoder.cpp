#include "encoder.hpp"

#include <cassert>
#include <cstdio>

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

void DefaultInit(KMeansParams* params) {
  assert(params);

  params->cluster_count = 10;
  params->attempt_count = 3;
  params->max_iter_count = 10;
  params->epsilon = 1;
}

void DefaultInit(RansacParams* params) {
  assert(params);

  params->subset_sz = 1;
  params->inlier_ratio = 0.5;
  params->success_prob = 0.99;
  params->inlier_thresh = 7.5;
}

void DefaultInit(EncoderConfig* cfg) {
  assert(cfg);

  cfg->mv_block_w = 16;
  cfg->mv_block_h = 16;
  cfg->mv_search_range = 8;
  cfg->pyr_lvl_count = 4;

  DefaultInit(&cfg->ransac);

  cfg->morph_rect_w = 3;
  cfg->morph_rect_h = 3;

  DefaultInit(&cfg->kmeans);

  cfg->connected_components_connectivity = 4;
  cfg->transform_block_w = 8;
  cfg->transform_block_h = 8;
}

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

Status Validate(KMeansParams* params) {
  assert(params);

  if (params->cluster_count == 0) {
    std::fprintf(stderr, "Cluster count must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (params->attempt_count == 0) {
    std::fprintf(stderr, "Attempt count must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (params->max_iter_count == 0) {
    std::fprintf(stderr, "Maximum iteration count must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (params->epsilon <= 0) {
    std::fprintf(stderr, "Epsilon must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  return kStatus_Ok;
}

Status Validate(RansacParams* params) {
  assert(params);

  if (params->inlier_thresh < 0) {
    std::fprintf(stderr, "Inlier threshold must be >= 0.\n");
    return kStatus_InvalidParamError;
  }

  if (params->success_prob < 0) {
    std::fprintf(stderr, "Success probability must be >= 0.\n");
    return kStatus_InvalidParamError;
  }

  if (params->inlier_ratio < 0) {
    std::fprintf(stderr, "Inlier ratio is >= 0.\n");
    return kStatus_InvalidParamError;
  }

  return kStatus_Ok;
}

Status Validate(EncoderConfig* cfg) {
  assert(cfg);

  if (cfg->mv_block_w < 1) {
    std::fprintf(stderr, "MV block width must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (cfg->mv_block_h < 1) {
    std::fprintf(stderr, "MV block height must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (cfg->pyr_lvl_count < 1) {
    std::fprintf(stderr, "Pyramid level count must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  uint top_lvl_reduction_factor = Pow2(cfg->pyr_lvl_count - 1);
  if (cfg->mv_search_range / top_lvl_reduction_factor == 0) {
    std::fprintf(stderr,
                 "The quotient from dividing the MV search range by the "
                 "top pyramid level reduction factor must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  Status status = Validate(&cfg->ransac);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "Failed to validate RANSAC parameters.\n");
    return status;
  }

  status = Validate(&cfg->kmeans);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "Failed to validate k-means parameters.\n");
    return status;
  }

  if (cfg->connected_components_connectivity != 4 &&
      cfg->connected_components_connectivity != 8) {
    std::fprintf(stderr,
                 "Connected components connectivity must be either 4 or 8.\n");
    return kStatus_InvalidParamError;
  }

  if (cfg->transform_block_w < 1) {
    std::fprintf(stderr, "Transform block width must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (cfg->transform_block_h < 1) {
    std::fprintf(stderr, "Transform block height must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  // IF the width and height of transform blocks are greater than the width
  // and height, respectively, of motion blocks, AND the width and height of
  // motion blocks are not divisible by the width and height, respectively, of
  // transform blocks, THEN the mapping of block types from motion blocks to
  // transform blocks would be ambiguous because a transform block would
  // overlap with multiple motion blocks.
  if (cfg->transform_block_w > cfg->mv_block_w) {
    std::fprintf(stderr, "Transform block width must be <= MV block width.\n");
    return kStatus_InvalidParamError;
  }
  if (cfg->transform_block_h > cfg->mv_block_h) {
    std::fprintf(stderr,
                 "Transform block height must be <= MV block height.\n");
    return kStatus_InvalidParamError;
  }
  if (cfg->mv_block_w % cfg->transform_block_w != 0) {
    std::fprintf(
        stderr, "MV block width must be divisible by transform block width.\n");
    return kStatus_InvalidParamError;
  }
  if (cfg->mv_block_h % cfg->transform_block_h != 0) {
    std::fprintf(
        stderr,
        "MV block height must be divisible by transform block height.\n");
    return kStatus_InvalidParamError;
  }

  return kStatus_Ok;
}

void InitEncoder(Encoder* enc, EncoderConfig* cfg) {
  assert(enc);
  assert(cfg);

  enc->cfg = *cfg;

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
  enc->padded_frame_w = ClosestLargerDivisible(cfg->frame_w, cfg->mv_block_w,
                                               top_lvl_reduction_factor);
  enc->padded_frame_h = ClosestLargerDivisible(cfg->frame_h, cfg->mv_block_h,
                                               top_lvl_reduction_factor);

  enc->mv_field_w = enc->padded_frame_w / cfg->mv_block_w;
  enc->mv_field_h = enc->padded_frame_h / cfg->mv_block_h;

  uint mv_field_sz = enc->mv_field_w * enc->mv_field_h;

  enc->mv_field.resize(mv_field_sz);
  enc->mv_field_min_mad.resize(mv_field_sz);

  enc->foreground_mv_field_mask.resize(mv_field_sz);
  enc->foreground_mv_field_indices.reserve(mv_field_sz);
  enc->foreground_mv_features.reserve(mv_field_sz);

  enc->mv_field_block_types.resize(mv_field_sz);

  enc->padded_frame = cv::Mat3b(enc->padded_frame_h, enc->padded_frame_w);
  enc->yuv_padded_frame = cv::Mat3b(enc->padded_frame_h, enc->padded_frame_w);
  enc->prev_y_padded_frame =
      cv::Mat1b(enc->padded_frame_h, enc->padded_frame_w);
  enc->y_padded_frame = cv::Mat1b(enc->padded_frame_h, enc->padded_frame_w);

  enc->prev_pyr.resize(cfg->pyr_lvl_count);
  enc->prev_pyr_data.resize(cfg->pyr_lvl_count);
  enc->pyr.resize(cfg->pyr_lvl_count);
  enc->pyr_data.resize(cfg->pyr_lvl_count);

  enc->prev_pyr[0] = enc->prev_y_padded_frame;
  enc->prev_pyr_data[0] = enc->prev_pyr[0].data;
  enc->pyr[0] = enc->y_padded_frame;
  enc->pyr_data[0] = enc->pyr[0].data;
  {
    uint w = enc->padded_frame_w / 2;
    uint h = enc->padded_frame_h / 2;
    for (uint i = 1; i < cfg->pyr_lvl_count; ++i) {
      enc->prev_pyr[i] = cv::Mat1b(h, w);
      enc->prev_pyr_data[i] = enc->prev_pyr[i].data;

      enc->pyr[i] = cv::Mat1b(h, w);
      enc->pyr_data[i] = enc->pyr[i].data;

      w /= 2;
      h /= 2;
    }
  }
}
