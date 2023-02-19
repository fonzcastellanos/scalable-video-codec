#include "motion.hpp"

#include <cassert>
#include <limits>
#include <random>

/**
 * @brief Calculates the mean absolute difference (MAD) between two blocks, each
 * from a different frame.
 *
 * A block's position is defined by the position of its top left corner.
 *
 * @param tracked_frame
 * @param anchor_frame
 * @param frame_w width of the frames
 * @param frame_h height of the frames
 * @param tracked_block_position
 * @param anchor_block_position
 * @param block_w width of the blocks
 * @param block_h height of the blocks
 * @return MAD
 */
static float BlockMad(const uchar* tracked_frame, const uchar* anchor_frame,
                      uint frame_w, uint frame_h, Vec2ui tracked_block_position,
                      Vec2ui anchor_block_position, uint block_w,
                      uint block_h) {
  assert(tracked_frame);
  assert(anchor_frame);

  float mad = 0;
  uint pixel_count = 1;

  for (uint y = 0; y < block_h; ++y) {
    for (uint x = 0; x < block_w; ++x) {
      uint ti = (tracked_block_position.y + y) * frame_w +
                tracked_block_position.x + x;

      uint ai =
          (anchor_block_position.y + y) * frame_w + anchor_block_position.x + x;

      uchar abs_diff = AbsDiff(tracked_frame[ti], anchor_frame[ai]);

      mad += (abs_diff - mad) / pixel_count;

      ++pixel_count;
    }
  }

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

      float mad =
          BlockMad(tracked_frame, anchor_frame, frame_w, frame_h,
                   tracked_block_pos, anchor_block_pos, block_w, block_h);

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
              BlockMad(tracked_frame, anchor_frame, frame_w, frame_h,
                       tracked_block_pos, anchor_block_pos, block_w, block_h);

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

static void RefineHierarchicalMotionEstimation(
    const uchar* tracked_frame, const uchar* anchor_frame, uint frame_w,
    uint frame_h, uint block_w, uint block_h, uint search_range,
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

  Vec2ui anchor_block_pos;
  for (uint mf_y = 0; mf_y < mfield_h; ++mf_y) {
    anchor_block_pos.y = mf_y * block_h;

    for (uint mf_x = 0; mf_x < mfield_w; ++mf_x) {
      anchor_block_pos.x = mf_x * block_w;

      uint mf_i = mf_y * mfield_w + mf_x;

      Vec2ui initial_tracked_block_pos = Vec2iToVec2ui(
          Vec2uiToVec2i(anchor_block_pos) + Vec2fToVec2i(motion_field[mf_i]));

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
              BlockMad(tracked_frame, anchor_frame, frame_w, frame_h,
                       tracked_block_pos, anchor_block_pos, block_w, block_h);

          if (mad < min_mad[mf_i]) {
            min_mad[mf_i] = mad;
            motion_field[mf_i] = Vec2iToVec2f(Vec2uiToVec2i(tracked_block_pos) -
                                              Vec2uiToVec2i(anchor_block_pos));
          }
        }
      }
    }
  }
}

void EstimateMotionHierarchical(const uchar* const* tracked_pyramid,
                                const uchar* const* anchor_pyramid,
                                uint num_levels, uint frame_w, uint frame_h,
                                uint search_range, uint block_w, uint block_h,
                                Vec2f* motion_field, float* min_mad) {
  assert(tracked_pyramid);
  assert(anchor_pyramid);
  assert(motion_field);
  assert(min_mad);

  assert(num_levels > 0);
  assert(block_w > 0);
  assert(block_h > 0);
  assert(frame_w % block_w == 0);
  assert(frame_h % block_h == 0);

  uint top_lvl_reduction_factor = Pow2(num_levels - 1);

  uint top_lvl_search_range = search_range / top_lvl_reduction_factor;

  assert(top_lvl_search_range > 0);

  uint fw = frame_w / top_lvl_reduction_factor;
  uint fh = frame_h / top_lvl_reduction_factor;

  uint bw = block_w / top_lvl_reduction_factor;
  uint bh = block_h / top_lvl_reduction_factor;

  EstimateMotionExhaustiveSearch(
      tracked_pyramid[num_levels - 1], anchor_pyramid[num_levels - 1], fw, fh,
      top_lvl_search_range, bw, bh, motion_field, min_mad);

  uint mfield_w = fw / bw;
  uint mfield_h = fh / bh;

  uint mfield_sz = mfield_w * mfield_h;

  for (int l = (int)num_levels - 2; l >= 0; --l) {
    fw *= 2;
    fh *= 2;

    bw *= 2;
    bh *= 2;

    for (uint i = 0; i < mfield_sz; ++i) {
      motion_field[i] *= 2.0f;
    }

    RefineHierarchicalMotionEstimation(tracked_pyramid[l], anchor_pyramid[l],
                                       fw, fh, bw, bh, top_lvl_search_range,
                                       motion_field, min_mad);
  }
}