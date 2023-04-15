#include "motion.hpp"

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <cassert>
#include <limits>
#include <random>

/*
Calculates the mean absolute difference (MAD) between two blocks, each from a
different frame.

Supports block areas <= 256.
*/
static float Mad(const uchar* a_frame, const uchar* b_frame, uint frame_w,
                 Vec2ui a_block_pos, Vec2ui b_block_pos, uint block_w,
                 uint block_h) {
  assert(a_frame);
  assert(b_frame);

  assert(frame_w > 0);
  assert(block_w > 0);
  assert(block_h > 0);

  uint sad = 0;

  for (uint k = 0; k < block_h; ++k) {
    for (uint j = 0; j < block_w; ++j) {
      uint ai = (a_block_pos.y + k) * frame_w + a_block_pos.x + j;
      uint bi = (b_block_pos.y + k) * frame_w + b_block_pos.x + j;

      sad += AbsDiff(a_frame[ai], b_frame[bi]);
    }
  }

  uint count = block_w * block_h;

  float mad = (float)sad / count;

  return mad;
}

Vec2f EstimateGlobalMotionAvg(const Vec2f* motion_field, uint sz) {
  assert(motion_field);

  Vec2f avg = {};
  for (uint i = 0; i < sz; ++i) {
    avg += (motion_field[i] - avg) * (1.0f / (i + 1));
  }
  return avg;
}

void EstimateGlobalMotionExhaustiveSearch(
    const uchar* tracked_frame, const uchar* anchor_frame, uint frame_w,
    uint frame_h, uint search_range, Vec2f* global_motion, float* min_mad) {
  assert(tracked_frame);
  assert(anchor_frame);
  assert(global_motion);
  assert(min_mad);

  assert(search_range <= frame_w);
  assert(search_range <= frame_h);

  *global_motion = {};
  *min_mad = std::numeric_limits<float>::max();

  Vec2ui tracked_block_pos;
  Vec2ui anchor_block_pos;

  for (int dy = -((int)search_range); dy <= search_range; ++dy) {
    uint tracked_block_begin_y = Max(0, dy);
    uint tracked_block_end_y = (int)frame_h + Min(0, dy);

    uint block_h = tracked_block_end_y - tracked_block_begin_y;

    tracked_block_pos.y = tracked_block_begin_y;
    anchor_block_pos.y = (int)tracked_block_pos.y - dy;

    for (int dx = -((int)search_range); dx <= search_range; ++dx) {
      uint tracked_block_begin_x = Max(0, dx);
      uint tracked_block_end_x = (int)frame_w + Min(0, dx);

      uint block_w = tracked_block_end_x - tracked_block_begin_x;

      tracked_block_pos.x = tracked_block_begin_x;
      anchor_block_pos.x = (int)tracked_block_pos.x - dx;

      float mad = Mad(tracked_frame, anchor_frame, frame_w, tracked_block_pos,
                      anchor_block_pos, block_w, block_h);

      if (mad < *min_mad) {
        *min_mad = mad;
        *global_motion = {(float)dx, (float)dy};
      }
    }
  }
}

void EstimateGlobalMotionHierarchical(const uchar* const* tracked_pyramid,
                                      const uchar* const* anchor_pyramid,
                                      uint num_levels, uint base_frame_w,
                                      uint base_frame_h, uint base_search_range,
                                      Vec2f* global_motion) {
  assert(tracked_pyramid);
  assert(anchor_pyramid);
  assert(global_motion);

  assert(num_levels != 0);
  assert(base_search_range <= base_frame_w);
  assert(base_search_range <= base_frame_h);

  uint top_lvl_reduction_factor = 1;
  for (uint i = 0; i < num_levels - 1; ++i) {
    top_lvl_reduction_factor <<= 1;
  }

  uint frame_w = base_frame_w / top_lvl_reduction_factor;
  uint frame_h = base_frame_h / top_lvl_reduction_factor;

  {
    // TODO: address last level search range becoming 0
    uint top_lvl_srch_range = base_search_range / top_lvl_reduction_factor;
    float min_mad;
    EstimateGlobalMotionExhaustiveSearch(
        tracked_pyramid[num_levels - 1], anchor_pyramid[num_levels - 1],
        frame_w, frame_h, top_lvl_srch_range, global_motion, &min_mad);
  }

  for (int l = (int)num_levels - 2; l >= 0; --l) {
    Vec2f corrective_motion;
    frame_h *= 2;
    frame_w *= 2;
    float min_mad;
    EstimateGlobalMotionExhaustiveSearch(tracked_pyramid[l], anchor_pyramid[l],
                                         frame_w, frame_h, 1,
                                         &corrective_motion, &min_mad);

    *global_motion = 2.0f * (*global_motion) + corrective_motion;
  }
}

static uint IterCount(RansacParams p) {
  float quot = Log(1 - p.success_prob);
  float div = Log(1 - Pow(p.inlier_ratio, p.subset_sz));
  uint res = Ceil(quot / div);
  return res;
}

static Vec2f AvgMotion(const Vec2f* motion_field, const uint* indices,
                       uint indices_sz) {
  assert(motion_field);
  assert(indices);

  Vec2f res = {};
  for (uint i = 0; i < indices_sz; ++i) {
    res += motion_field[indices[i]];
  }
  res = res * (1.0f / indices_sz);

  return res;
}

static float Rmse(const Vec2f* motion_field, const uint* indices,
                  uint indices_sz, Vec2f estimated_motion) {
  assert(motion_field);
  assert(indices);

  float res = 0;

  for (uint i = 0; i < indices_sz; ++i) {
    Vec2f mv = motion_field[indices[i]];
    res += Sqr(mv.x - estimated_motion.x) + Sqr(mv.y - estimated_motion.y);
  }

  res = Sqrt(res / indices_sz);

  return res;
}

void EstimateGlobalMotionRansac(const Vec2f* motion_field, uint motion_field_sz,
                                RansacParams params, float* rmse,
                                Vec2f* global_motion,
                                std::vector<uint>* inliers) {
  static std::random_device rdev;
  static std::default_random_engine reng(rdev());

  assert(motion_field);
  assert(rmse);
  assert(global_motion);
  assert(inliers);

  assert(motion_field_sz >= params.subset_sz);

  uint iter_count = IterCount(params);

  std::vector<uint> subset(params.subset_sz);
  std::vector<uint> best_subset(params.subset_sz);

  std::vector<uint> inliers_;
  inliers_.reserve(motion_field_sz);
  std::vector<uint> best_inliers;
  best_inliers.reserve(motion_field_sz);

  Vec2f best_glob_motion;

  std::uniform_int_distribution<uint> distrib(0, motion_field_sz);

  for (uint iter = 0; iter < iter_count; ++iter) {
    for (uint i = 0; i < params.subset_sz; ++i) {
      uint j;
      do {
        j = 0;
        subset[i] = distrib(reng);
        while (j < i && subset[j] != subset[i]) {
          ++j;
        }
      } while (j < i);
    }

    Vec2f gm = AvgMotion(motion_field, subset.data(), subset.size());

    inliers_.clear();
    for (uint i = 0; i < motion_field_sz; ++i) {
      Vec2f m = motion_field[i];

      if (Sqr(gm.x - m.x) + Sqr(gm.y - m.y) < Sqr(params.inlier_thresh)) {
        inliers_.push_back(i);
      }
    }

    if (inliers_.size() >= best_inliers.size()) {
      best_glob_motion = gm;
      best_subset.swap(subset);
      best_inliers.swap(inliers_);
    }
  }

  if (best_inliers.size() < params.subset_sz) {
    *rmse = Rmse(motion_field, best_subset.data(), params.subset_sz,
                 *global_motion);
  } else {
    inliers_.clear();
    for (uint i = 0;
         i < motion_field_sz && inliers_.size() < best_inliers.size(); ++i) {
      Vec2f m = motion_field[i];

      if (Sqr(best_glob_motion.x - m.x) + Sqr(best_glob_motion.y - m.y) <
          Sqr(params.inlier_thresh)) {
        inliers_.push_back(i);
      }
    }

    best_glob_motion =
        AvgMotion(motion_field, inliers_.data(), inliers_.size());

    *rmse =
        Rmse(motion_field, inliers_.data(), inliers_.size(), best_glob_motion);

    best_inliers.swap(inliers_);
  }

  *global_motion = best_glob_motion;
  inliers->swap(best_inliers);
}

void EstimateMotionExhaustiveSearch(const uchar* tracked_frame,
                                    const uchar* anchor_frame, uint frame_w,
                                    uint frame_h, uint search_range,
                                    uint block_w, uint block_h,
                                    Vec2f* motion_field, float* min_mad) {
  assert(tracked_frame);
  assert(anchor_frame);
  assert(motion_field);
  assert(min_mad);

  assert(block_w > 0);
  assert(block_h > 0);

  assert(frame_w % block_w == 0);
  assert(frame_h % block_h == 0);

  uint mfield_w = frame_w / block_w;
  uint mfield_h = frame_h / block_h;

  uint mfield_sz = mfield_w * mfield_h;
  for (uint i = 0; i < mfield_sz; ++i) {
    motion_field[i] = {};
    min_mad[i] = std::numeric_limits<float>::max();
  }

  Vec2ui anchor_block_pos;
  for (uint mf_y = 0; mf_y < mfield_h; ++mf_y) {
    anchor_block_pos.y = mf_y * block_h;

    uint srch_begin_y = Max(0, (int)anchor_block_pos.y - (int)search_range);
    uint srch_end_y =
        Min(frame_h - block_h + 1, anchor_block_pos.y + search_range + 1);

    uint srch_h = srch_end_y - srch_begin_y;

    for (uint mf_x = 0; mf_x < mfield_w; ++mf_x) {
      anchor_block_pos.x = mf_x * block_w;

      uint mf_i = mf_y * mfield_w + mf_x;

      uint srch_begin_x = Max(0, (int)anchor_block_pos.x - (int)search_range);
      uint srch_end_x =
          Min(frame_w - block_w + 1, anchor_block_pos.x + search_range + 1);

      uint min_mad_update_count = 0;
      Vec2ui tracked_block_pos;
      for (uint y = srch_begin_y; y < srch_end_y; ++y) {
        tracked_block_pos.y = y;

        for (uint x = srch_begin_x; x < srch_end_x; ++x) {
          tracked_block_pos.x = x;

          float mad =
              Mad(tracked_frame, anchor_frame, frame_w, tracked_block_pos,
                  anchor_block_pos, block_w, block_h);

          if (mad <= min_mad[mf_i]) {
            min_mad[mf_i] = mad;
            motion_field[mf_i] = Vec2iToVec2f(Vec2uiToVec2i(tracked_block_pos) -
                                              Vec2uiToVec2i(anchor_block_pos));
            ++min_mad_update_count;
          }
        }
      }

      uint srch_w = srch_end_x - srch_begin_x;
      uint srch_area = srch_h * srch_w;
      if (min_mad_update_count == srch_area) {
        motion_field[mf_i] = {};
      }
    }
  }
}

static void RefineHierMotionEst(const uchar* tracked_frame,
                                const uchar* anchor_frame, uint frame_w,
                                uint frame_h, uint block_w, uint block_h,
                                uint search_range, Vec2f* mv_field,
                                float* min_mad) {
  assert(tracked_frame);
  assert(anchor_frame);
  assert(mv_field);
  assert(min_mad);

  assert(frame_w > 0);
  assert(frame_h > 0);
  assert(block_w > 0);
  assert(block_h > 0);

  assert(frame_w % block_w == 0);
  assert(frame_h % block_h == 0);

  uint mv_field_w = frame_w / block_w;
  uint mv_field_h = frame_h / block_h;

  Vec2ui anchor_block_pos;
  for (uint mvf_y = 0; mvf_y < mv_field_h; ++mvf_y) {
    anchor_block_pos.y = mvf_y * block_h;

    for (uint mvf_x = 0; mvf_x < mv_field_w; ++mvf_x) {
      anchor_block_pos.x = mvf_x * block_w;

      uint mvf_i = mvf_y * mv_field_w + mvf_x;

      Vec2ui initial_tracked_block_pos = Vec2iToVec2ui(
          Vec2uiToVec2i(anchor_block_pos) + Vec2fToVec2i(mv_field[mvf_i]));

      uint srch_region_begin_y =
          Max(0, (int)initial_tracked_block_pos.y - (int)search_range);
      uint srch_region_end_y =
          Min(frame_h - block_h + 1,
              initial_tracked_block_pos.y + search_range + 1);

      uint srch_region_begin_x =
          Max(0, (int)initial_tracked_block_pos.x - (int)search_range);
      uint srch_region_end_x =
          Min(frame_w - block_w + 1,
              initial_tracked_block_pos.x + search_range + 1);

      // TODO: Should I skip calculating mad for block at (tracked_x,
      // tracked_y)?
      //  Is it redundant?
      Vec2ui tracked_block_pos;
      for (uint y = srch_region_begin_y; y < srch_region_end_y; ++y) {
        tracked_block_pos.y = y;

        for (uint x = srch_region_begin_x; x < srch_region_end_x; ++x) {
          tracked_block_pos.x = x;

          float mad =
              Mad(tracked_frame, anchor_frame, frame_w, tracked_block_pos,
                  anchor_block_pos, block_w, block_h);

          if (mad < min_mad[mvf_i]) {
            min_mad[mvf_i] = mad;
            mv_field[mvf_i] = Vec2iToVec2f(Vec2uiToVec2i(tracked_block_pos) -
                                           Vec2uiToVec2i(anchor_block_pos));
          }
        }
      }
    }
  }
}

void EstimateMotionHierarchical(const uchar* const* tracked_pyramid,
                                const uchar* const* anchor_pyramid,
                                uint level_count, uint frame_w, uint frame_h,
                                uint search_range, uint block_w, uint block_h,
                                Vec2f* motion_field, float* min_mad) {
  assert(tracked_pyramid);
  assert(anchor_pyramid);
  assert(motion_field);
  assert(min_mad);

  assert(level_count > 0);
  assert(block_w > 0);
  assert(block_h > 0);
  assert(frame_w > 0);
  assert(frame_h > 0);

  assert(frame_w % block_w == 0);
  assert(frame_h % block_h == 0);

  uint top_lvl_reduction_factor = Pow2(level_count - 1);

  assert(search_range >= top_lvl_reduction_factor);

  uint top_lvl_srch_range = search_range / top_lvl_reduction_factor;

  uint fw = frame_w / top_lvl_reduction_factor;
  uint fh = frame_h / top_lvl_reduction_factor;

  uint bw = block_w / top_lvl_reduction_factor;
  uint bh = block_h / top_lvl_reduction_factor;

  EstimateMotionExhaustiveSearch(
      tracked_pyramid[level_count - 1], anchor_pyramid[level_count - 1], fw, fh,
      top_lvl_srch_range, bw, bh, motion_field, min_mad);

  uint mv_field_w = frame_w / block_w;
  uint mv_field_h = frame_h / block_h;
  uint mv_field_sz = mv_field_w * mv_field_h;

  for (int l = (int)level_count - 2; l >= 0; --l) {
    fw *= 2;
    fh *= 2;

    bw *= 2;
    bh *= 2;

    for (uint i = 0; i < mv_field_sz; ++i) {
      motion_field[i] *= 2.0f;
    }

    RefineHierMotionEst(tracked_pyramid[l], anchor_pyramid[l], fw, fh, bw, bh,
                        top_lvl_srch_range, motion_field, min_mad);
  }
}

#ifdef __SSE2__
/*
Calculates the mean absolute difference (MAD) between two 16x16 blocks, each
from a different frame.
*/
static float Mad16x16Sse2(const uchar* a_frame, const uchar* b_frame,
                          uint frame_w, Vec2ui a_block_pos,
                          Vec2ui b_block_pos) {
  static constexpr uint kBlockW = 16;
  static constexpr uint kBlockH = 16;
  static constexpr float kBlockArea = kBlockW * kBlockH;

  assert(a_frame);
  assert(b_frame);

  assert(frame_w >= kBlockW);

  __m128i sum1 = _mm_setzero_si128();
  __m128i sum2 = _mm_setzero_si128();

  for (uint i = 0; i < kBlockH; i += 2) {
    __m128i a1 = _mm_loadu_si128(
        (__m128i*)(a_frame + (a_block_pos.y + i) * frame_w + a_block_pos.x));
    __m128i b1 = _mm_loadu_si128(
        (__m128i*)(b_frame + (b_block_pos.y + i) * frame_w + b_block_pos.x));

    __m128i a2 = _mm_loadu_si128((
        __m128i*)(a_frame + (a_block_pos.y + i + 1) * frame_w + a_block_pos.x));
    __m128i b2 = _mm_loadu_si128((
        __m128i*)(b_frame + (b_block_pos.y + i + 1) * frame_w + b_block_pos.x));

    sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(a1, b1));
    sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(a2, b2));
  }

  sum1 = _mm_add_epi64(sum1, sum2);

  long long sad =
      _mm_cvtsi128_si64(_mm_add_epi64(sum1, _mm_srli_si128(sum1, 8)));

  float mad = sad / kBlockArea;

  return mad;
}

/*
Calculates the mean absolute difference (MAD) between two 8x8 blocks, each
from a different frame.
*/
static float Mad8x8Sse2(const uchar* a_frame, const uchar* b_frame,
                        uint frame_w, Vec2ui a_block_pos, Vec2ui b_block_pos) {
  static constexpr uint kBlockW = 8;
  static constexpr uint kBlockH = 8;
  static constexpr float kBlockArea = kBlockW * kBlockH;

  assert(a_frame);
  assert(b_frame);

  assert(frame_w >= kBlockW);

  __m128i sum = _mm_setzero_si128();

  for (uint i = 0; i < kBlockH; i += 2) {
    __m128i a0 = _mm_loadu_si64(
        (__m128i*)(a_frame + (a_block_pos.y + i) * frame_w + a_block_pos.x));
    __m128i a1 = _mm_loadu_si64((
        __m128i*)(a_frame + (a_block_pos.y + i + 1) * frame_w + a_block_pos.x));
    __m128i a = _mm_unpacklo_epi8(a0, a1);

    __m128i b0 = _mm_loadu_si64(
        (__m128i*)(b_frame + (b_block_pos.y + i) * frame_w + b_block_pos.x));
    __m128i b1 = _mm_loadu_si64((
        __m128i*)(b_frame + (b_block_pos.y + i + 1) * frame_w + b_block_pos.x));
    __m128i b = _mm_unpacklo_epi8(b0, b1);

    sum = _mm_add_epi64(sum, _mm_sad_epu8(a, b));
  }

  long long sad = _mm_cvtsi128_si64(_mm_add_epi64(sum, _mm_srli_si128(sum, 8)));

  float mad = sad / kBlockArea;

  return mad;
}

static void RefineHierMotionEst16x16Sse2(const uchar* tracked_frame,
                                         const uchar* anchor_frame,
                                         uint frame_w, uint frame_h,
                                         uint search_range, Vec2f* mv_field,
                                         float* min_mad) {
  static constexpr uint kBlockW = 16;
  static constexpr uint kBlockH = 16;

  assert(tracked_frame);
  assert(anchor_frame);
  assert(mv_field);
  assert(min_mad);

  assert(frame_w > 0);
  assert(frame_h > 0);

  assert(frame_w % kBlockW == 0);
  assert(frame_h % kBlockH == 0);

  uint mv_field_w = frame_w / kBlockW;
  uint mv_field_h = frame_h / kBlockH;

  Vec2ui anchor_block_pos;
  for (uint mvf_y = 0; mvf_y < mv_field_h; ++mvf_y) {
    anchor_block_pos.y = mvf_y * kBlockH;

    for (uint mvf_x = 0; mvf_x < mv_field_w; ++mvf_x) {
      anchor_block_pos.x = mvf_x * kBlockW;

      uint mvf_i = mvf_y * mv_field_w + mvf_x;

      Vec2ui initial_tracked_block_pos = Vec2iToVec2ui(
          Vec2uiToVec2i(anchor_block_pos) + Vec2fToVec2i(mv_field[mvf_i]));

      uint srch_region_begin_y =
          Max(0, (int)initial_tracked_block_pos.y - (int)search_range);
      uint srch_region_end_y =
          Min(frame_h - kBlockH + 1,
              initial_tracked_block_pos.y + search_range + 1);

      uint srch_region_begin_x =
          Max(0, (int)initial_tracked_block_pos.x - (int)search_range);
      uint srch_region_end_x =
          Min(frame_w - kBlockW + 1,
              initial_tracked_block_pos.x + search_range + 1);

      // TODO: Should I skip calculating mad for block at (tracked_x,
      // tracked_y)?
      //  Is it redundant?
      Vec2ui tracked_block_pos;
      for (uint y = srch_region_begin_y; y < srch_region_end_y; ++y) {
        tracked_block_pos.y = y;

        for (uint x = srch_region_begin_x; x < srch_region_end_x; ++x) {
          tracked_block_pos.x = x;

          float mad = Mad16x16Sse2(tracked_frame, anchor_frame, frame_w,
                                   tracked_block_pos, anchor_block_pos);

          if (mad < min_mad[mvf_i]) {
            min_mad[mvf_i] = mad;
            mv_field[mvf_i] = Vec2iToVec2f(Vec2uiToVec2i(tracked_block_pos) -
                                           Vec2uiToVec2i(anchor_block_pos));
          }
        }
      }
    }
  }
}

static void RefineHierMotionEst8x8Sse2(const uchar* tracked_frame,
                                       const uchar* anchor_frame, uint frame_w,
                                       uint frame_h, uint search_range,
                                       Vec2f* mv_field, float* min_mad) {
  static constexpr uint kBlockW = 8;
  static constexpr uint kBlockH = 8;

  assert(tracked_frame);
  assert(anchor_frame);
  assert(mv_field);
  assert(min_mad);

  assert(frame_w > 0);
  assert(frame_h > 0);

  assert(frame_w % kBlockW == 0);
  assert(frame_h % kBlockH == 0);

  uint mv_field_w = frame_w / kBlockW;
  uint mv_field_h = frame_h / kBlockH;

  Vec2ui anchor_block_pos;
  for (uint mvf_y = 0; mvf_y < mv_field_h; ++mvf_y) {
    anchor_block_pos.y = mvf_y * kBlockH;

    for (uint mvf_x = 0; mvf_x < mv_field_w; ++mvf_x) {
      anchor_block_pos.x = mvf_x * kBlockW;

      uint mvf_i = mvf_y * mv_field_w + mvf_x;

      Vec2ui initial_tracked_block_pos = Vec2iToVec2ui(
          Vec2uiToVec2i(anchor_block_pos) + Vec2fToVec2i(mv_field[mvf_i]));

      uint srch_region_begin_y =
          Max(0, (int)initial_tracked_block_pos.y - (int)search_range);
      uint srch_region_end_y =
          Min(frame_h - kBlockH + 1,
              initial_tracked_block_pos.y + search_range + 1);

      uint srch_region_begin_x =
          Max(0, (int)initial_tracked_block_pos.x - (int)search_range);
      uint srch_region_end_x =
          Min(frame_w - kBlockW + 1,
              initial_tracked_block_pos.x + search_range + 1);

      // TODO: Should I skip calculating mad for block at (tracked_x,
      // tracked_y)?
      //  Is it redundant?
      Vec2ui tracked_block_pos;
      for (uint y = srch_region_begin_y; y < srch_region_end_y; ++y) {
        tracked_block_pos.y = y;

        for (uint x = srch_region_begin_x; x < srch_region_end_x; ++x) {
          tracked_block_pos.x = x;

          float mad = Mad8x8Sse2(tracked_frame, anchor_frame, frame_w,
                                 tracked_block_pos, anchor_block_pos);

          if (mad < min_mad[mvf_i]) {
            min_mad[mvf_i] = mad;
            mv_field[mvf_i] = Vec2iToVec2f(Vec2uiToVec2i(tracked_block_pos) -
                                           Vec2uiToVec2i(anchor_block_pos));
          }
        }
      }
    }
  }
}

void EstimateMotionHierarchical16x16Sse2(const uchar* const* tracked_pyramid,
                                         const uchar* const* anchor_pyramid,
                                         uint frame_w, uint frame_h,
                                         uint search_range, Vec2f* mv_field,
                                         float* min_mad) {
  static constexpr uint kLevelCount = 4;
  static constexpr uint kBlockW = 16;
  static constexpr uint kBlockH = 16;
  static constexpr uint kTopLevelReductionFactor = 1 << (kLevelCount - 1);

  assert(tracked_pyramid);
  assert(anchor_pyramid);
  assert(mv_field);
  assert(min_mad);

  assert(frame_w > 0);
  assert(frame_h > 0);

  assert(frame_w % kBlockW == 0);
  assert(frame_h % kBlockH == 0);

  assert(search_range >= kTopLevelReductionFactor);

  uint top_lvl_srch_range = search_range / kTopLevelReductionFactor;

  uint fw = frame_w / kTopLevelReductionFactor;
  uint fh = frame_h / kTopLevelReductionFactor;

  EstimateMotionExhaustiveSearch(tracked_pyramid[3], anchor_pyramid[3], fw, fh,
                                 top_lvl_srch_range, 2, 2, mv_field, min_mad);

  uint mv_field_w = frame_w / kBlockW;
  uint mv_field_h = frame_h / kBlockH;
  uint mv_field_sz = mv_field_w * mv_field_h;

  for (uint i = 0; i < mv_field_sz; ++i) {
    mv_field[i] *= 2.0f;
  }
  fw *= 2;
  fh *= 2;
  RefineHierMotionEst(tracked_pyramid[2], anchor_pyramid[2], fw, fh, 4, 4,
                      top_lvl_srch_range, mv_field, min_mad);

  for (uint i = 0; i < mv_field_sz; ++i) {
    mv_field[i] *= 2.0f;
  }
  fw *= 2;
  fh *= 2;
  RefineHierMotionEst8x8Sse2(tracked_pyramid[1], anchor_pyramid[1], fw, fh,
                             top_lvl_srch_range, mv_field, min_mad);

  for (uint i = 0; i < mv_field_sz; ++i) {
    mv_field[i] *= 2.0f;
  }
  fw *= 2;
  fh *= 2;
  RefineHierMotionEst16x16Sse2(tracked_pyramid[0], anchor_pyramid[0], fw, fh,
                               top_lvl_srch_range, mv_field, min_mad);
}
#endif