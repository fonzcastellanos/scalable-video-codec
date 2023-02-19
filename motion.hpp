#ifndef SCALABLE_VIDEO_CODEC_MOTION_HPP
#define SCALABLE_VIDEO_CODEC_MOTION_HPP

#include <vector>

#include "math.hpp"
#include "types.hpp"

/*
Motion estimation functions.

Abbreviations / Acronyms:
- MV = motion vector
- EBMA = exhaustive search block matching algorithm
- HBMA = hierarchhical block matching algorithm

A block-wise translational (i.e. 2D) motion model is assumed.

All functions dedicated to motion estimation exclusively perform backwards
motion estimation, where the anchor frame is in the temporal posterior to the
tracked frame, and a motion vector is from an anchor frame to a tracked frame.

The motion estimation criterion involves finding the tracked frame block
within the search region for each anchor frame block, with the aim of
minimizing the mean absolute difference (MAD) between the two blocks.

The search range parameter R describes a region (i.e. the search region) in the
tracked frame that is symmetric with respect to the current anchor block, up to
R pixels to the left and right, and up to R pixels above and below. This region
encompasses the candidate blocks from the tracked frame that are compared to the
current block in the anchor frame.
*/

/*
Estimates global motion by calculating the average MV from the motion field.
*/
Vec2f EstimateGlobalMotionAvg(const Vec2f* motion_field, uint sz);

/*
Estimates global motion using EBMA.

The entire anchor frame is used as the template.
*/
void EstimateGlobalMotionExhaustiveSearch(const uchar* tracked_frame,
                                          const uchar* anchor_frame,
                                          uint frame_w, uint frame_h,
                                          uint search_range,
                                          Vec2f* global_motion, float* min_mad);

/* Estimates global motion using a HBMA.

The entire anchor frame is used as the template.
*/
void EstimateGlobalMotionHierarchical(const uchar* const* tracked_pyramid,
                                      const uchar* const* anchor_pyramid,
                                      uint num_levels, uint base_frame_w,
                                      uint base_frame_h, uint base_search_range,
                                      Vec2f* global_motion);

struct RansacParams {
  /*
  Minimum number of data points to estimate the underlying model parameters.
  */
  uint subset_sz;

  /*
  Error threshold for classifying a data point as an inlier.
  */
  float inlier_thresh;

  /*
  The probability that at least one of the sets of random samples does not
  include an outlier.
  */
  float success_prob;

  /* The probability that any selected data point is an inlier. */
  float inlier_ratio;
};

/*
Estimates the global motion using RANSAC.

Fitting is done by taking the average of a sampled subset.


A motion vector is classified as an inlier if the sum of the squared errors
is less than the squared threshold:

(estimated_mv.x - mv.x)^2 + (estimated_mv.y - mv.y)^2 < `inlier_thresh`^2


The number of iterations k is determined by the following equation:

k = log(1 - p) / (1 - w ^ n)

p is the desired probability of success, w is the probability that any selected
data point is an inlier (i.e. inlier ratio), and n is the subset size.
*/
void EstimateGlobalMotionRansac(const Vec2f* motion_field, uint motion_field_sz,
                                RansacParams params, float* rmse,
                                Vec2f* global_motion,
                                std::vector<uint>* inlier_indices);

/* Calculates the motion field using EBMA. */
void EstimateMotionExhaustiveSearch(const uchar* tracked_frame,
                                    const uchar* anchor_frame, uint frame_w,
                                    uint frame_h, uint search_range,
                                    uint block_w, uint block_h,
                                    Vec2f* motion_field, float* min_mad);

/*
Calculates the motion field using a variation of HBMA.

The initial MVs are searched at the top pyramid level with a search range of
`search_range` / 2 ^ (`level_count` - 1). At each of the remaining levels,
correction vectors are searched with the same search range as that of the top
level.

At each increasing level of the input pyramids, the frame dimensions are
assumed to be halved.

At each increasing level, the block dimensions are halved,
starting from those of the base level provided as parameters: `block_w`
and `block_h`.

The frame and block dimension parameters are of the base pyramid levels.
*/
void EstimateMotionHierarchical(const uchar* const* tracked_pyramid,
                                const uchar* const* anchor_pyramid,
                                uint level_count, uint frame_w, uint frame_h,
                                uint search_range, uint block_w, uint block_h,
                                Vec2f* motion_field, float* min_mad);

#endif  // SCALABLE_VIDEO_CODEC_MOTION_HPP