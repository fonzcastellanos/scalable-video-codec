#include "encoder_config.hpp"

#include <cassert>
#include <cstdio>

#include "cli.hpp"

Status ParseConfig(uint argc, char* argv[], Config* c) {
  assert(argv);
  assert(c);

  EncoderConfig* ec = &c->encoder;

  Status res = kStatus_InvalidParamError;

  /*******************************************************************************
   * Command-line Options    #options
   *******************************************************************************/
  Option opts[] {
#if !defined(__SSE2__) || !defined(SVC_MOTION_SSE2)
    {"mv-block-w", kOptionTypeUInt, &ec->mv_block_w},
        {"mv-block-h", kOptionTypeUInt, &ec->mv_block_h},
        {"pyr-lvl-count", kOptionTypeUInt, &ec->pyr_lvl_count},
#endif
        {"mv-search-range", kOptionTypeUInt, &ec->mv_search_range},
        {"ransac-subset-sz", kOptionTypeUInt, &ec->ransac.subset_sz},
        {"ransac-inlier-thresh", kOptionTypeFloat, &ec->ransac.inlier_thresh},
        {"ransac-success-prob", kOptionTypeFloat, &ec->ransac.success_prob},
        {"ransac-inlier-ratio", kOptionTypeFloat, &ec->ransac.inlier_ratio},
        {"morph-rect-w", kOptionTypeUInt, &ec->morph_rect_w},
        {"morph-rect-h", kOptionTypeUInt, &ec->morph_rect_h},
        {"kmeans-cluster-count", kOptionTypeUInt, &ec->kmeans.cluster_count},
        {"kmeans-attempt-count", kOptionTypeUInt, &ec->kmeans.attempt_count},
        {"kmeans-max-iter-count", kOptionTypeUInt, &ec->kmeans.max_iter_count},
        {"kmeans-epsilon", kOptionTypeFloat, &ec->kmeans.epsilon},
        {"connected-components-connectivity", kOptionTypeUInt,
         &ec->connected_components_connectivity},
        {"transform-block-w", kOptionTypeUInt, &ec->transform_block_w},
        {"transform-block-h", kOptionTypeUInt, &ec->transform_block_h}, {
      "verbose", kOptionTypeBoolean, &c->verbose
    }
  };

  uint argi;
  uint opts_size = sizeof(opts) / sizeof(opts[0]);

  CliStatus st = ParseOptions(argc, argv, opts_size, opts, &argi);
  if (st != kCliStatusOk) {
    std::fprintf(stderr, "parsing options: %s\n", CliStatusMessage(st));
    return res;
  }

  if (argc < argi + 1) {
    std::fprintf(stderr, "missing video path argument\n");
    return res;
  }

  c->video_path = argv[argi];

  res = kStatus_Ok;

  return res;
}

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

void DefaultInit(Config* c) {
  assert(c);

  DefaultInit(&c->encoder);

  c->video_path = 0;
  c->verbose = 1;
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
