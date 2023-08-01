#include "encoder.hpp"

#include <cassert>
#include <cstdio>
#include <opencv2/core.hpp>
#include <utility>
#ifdef VISUALIZE
#include <opencv2/highgui.hpp>
#endif  // VISUALIZE
#include <opencv2/imgproc.hpp>

#ifdef VISUALIZE
#include "draw.hpp"
#endif  // VISUALIZE

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

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

Encoder::Encoder(const EncoderConfig& cfg, const VideoProperties& vidprops,
                 CircularQueue<cv::Mat3b>& in_queue,
                 std::future<void> attempted_first_frame_read,
                 CircularQueue<std::vector<uchar>>& out_queue)
    : cfg_{cfg},
      vidprops_{vidprops},
      in_queue_{in_queue},
      attempted_first_frame_read_{std::move(attempted_first_frame_read)},
      out_queue_{out_queue} {
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
  uint top_lvl_reduction_factor = Pow2(cfg_.pyr_lvl_count - 1);
  padded_frame_w_ = ClosestLargerDivisible(vidprops_.frame_w, cfg_.mv_block_w,
                                           top_lvl_reduction_factor);
  padded_frame_h_ = ClosestLargerDivisible(vidprops_.frame_h, cfg_.mv_block_h,
                                           top_lvl_reduction_factor);

  frame_excess_w_ = padded_frame_w_ - vidprops_.frame_w;
  frame_excess_h_ = padded_frame_h_ - vidprops_.frame_h;

  mv_field_w_ = padded_frame_w_ / cfg_.mv_block_w;
  mv_field_h_ = padded_frame_h_ / cfg_.mv_block_h;

  uint mv_field_sz = mv_field_w_ * mv_field_h_;

  mv_field_.resize(mv_field_sz);
  mv_field_min_mad_.resize(mv_field_sz);

  foreground_mv_field_mask_ = cv::Mat1b(mv_field_h_, mv_field_w_);
  foreground_mv_field_indices_.reserve(mv_field_sz);
  foreground_mv_features_.reserve(mv_field_sz);
  mv_field_block_types_.resize(mv_field_sz);

  foreground_cluster_mask_(mv_field_h_, mv_field_w_);

  morph_rect_ = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(cfg_.morph_rect_w, cfg_.morph_rect_h));

  padded_frame_ = cv::Mat3b(padded_frame_h_, padded_frame_w_);
  yuv_padded_frame_ = cv::Mat3b(padded_frame_h_, padded_frame_w_);
  prev_y_padded_frame_ = cv::Mat1b(padded_frame_h_, padded_frame_w_);
  y_padded_frame_ = cv::Mat1b(padded_frame_h_, padded_frame_w_);

  prev_pyr_.resize(cfg_.pyr_lvl_count);
  prev_pyr_data_.resize(cfg_.pyr_lvl_count);
  pyr_.resize(cfg_.pyr_lvl_count);
  pyr_data_.resize(cfg_.pyr_lvl_count);

  prev_pyr_[0] = prev_y_padded_frame_;
  prev_pyr_data_[0] = prev_pyr_[0].data;
  pyr_[0] = y_padded_frame_;
  pyr_data_[0] = pyr_[0].data;
  {
    uint w = padded_frame_w_ / 2;
    uint h = padded_frame_h_ / 2;
    for (uint i = 1; i < cfg_.pyr_lvl_count; ++i) {
      prev_pyr_[i] = cv::Mat1b(h, w);
      prev_pyr_data_[i] = prev_pyr_[i].data;

      pyr_[i] = cv::Mat1b(h, w);
      pyr_data_[i] = pyr_[i].data;

      w /= 2;
      h /= 2;
    }
  }
}

static std::vector<uchar> SerializeEncodedFrame(
    const std::vector<cv::Mat1f>* dct_coeffs,
    const std::vector<uint>* mv_field_block_types, uint frame_w, uint frame_h,
    uint transform_block_w, uint transform_block_h, uint mv_field_w,
    uint mv_field_h, uint mv_block_w, uint mv_block_h) {
  assert(dct_coeffs);
  assert(mv_field_block_types);

  assert(transform_block_h > 0);
  assert(transform_block_w > 0);

  assert(frame_w % transform_block_h == 0);
  assert(frame_h % transform_block_w == 0);

  assert(mv_field_w * mv_field_h == mv_field_block_types->size());

  assert(transform_block_h <= mv_block_w);
  assert(transform_block_w <= mv_block_h);

  assert(mv_block_h % transform_block_w == 0);
  assert(mv_block_w % transform_block_h == 0);

  std::vector<uchar> result;

  for (uint tb_y = 0; tb_y < frame_h; tb_y += transform_block_h) {
    for (uint tb_x = 0; tb_x < frame_w; tb_x += transform_block_w) {
      uint mv_field_y = tb_y / mv_block_h;
      uint mv_field_x = tb_x / mv_block_w;
      uint mv_field_i = mv_field_y * mv_field_w + mv_field_x;

      uint btype = (*mv_field_block_types)[mv_field_i];

      result.insert(result.end(), (uchar*)(&btype),
                    ((uchar*)(&btype)) + sizeof(btype));

      for (const auto& channel : (*dct_coeffs)) {
        float* coeffs = (float*)channel.data;

        for (uint y = tb_y; y < tb_y + transform_block_w; ++y) {
          float* row = &coeffs[y * frame_w + tb_x];

          result.insert(result.end(), (uchar*)row,
                        ((uchar*)row) + sizeof(float) * transform_block_h);
        }
      }
    }
  }

  return result;
}

#ifdef VISUALIZE
const char* kWindowName = "Encoding";

struct ViewTitleTextParams {
  cv::Scalar outline_color;
  cv::Scalar fill_color;
  cv::HersheyFonts font;
  float font_scale_factor;
  cv::Point2i origin;
  cv::LineTypes line_type;
  float line_thickness_scale_factor;
};

static void DrawViewTitle(cv::Mat* frame, const char* text,
                          ViewTitleTextParams* params) {
  assert(frame);
  assert(text);
  assert(params);

  DrawOutlinedText(frame, text, params->origin, params->outline_color,
                   params->fill_color, params->font, params->font_scale_factor,
                   params->line_thickness_scale_factor, params->line_type);
}
#endif  // VISUALIZE

static void BuildMvFeatures(const Vec2f* mv_field, uint mv_field_w,
                            uint mv_block_w, uint mv_block_h,
                            const uint* mv_field_indices,
                            uint mv_field_indices_sz, Vec4f* features) {
  assert(mv_field);
  assert(features);
  assert(mv_field_w > 0);
  assert(mv_field_indices_sz == 0 || mv_field_indices);

  for (uint i = 0; i < mv_field_indices_sz; ++i) {
    uint mf_i = mv_field_indices[i];

    uint mf_y = mf_i / mv_field_w;
    uint mf_x = mf_i % mv_field_w;

    uint y = mf_y * mv_block_h;
    uint x = mf_x * mv_block_w;

    Vec2f m = mv_field[mf_i];

    features[i][0] = m.x;
    features[i][1] = m.y;
    features[i][1] = x;
    features[i][2] = y;
  }
}

static void Dct(const cv::Mat3f* frame, uint block_w, uint block_h,
                std::vector<cv::Mat1f>* coeffs) {
  assert(frame);
  assert(coeffs);

  assert(block_h > 0);
  assert(block_w > 0);

  cv::split(*frame, *coeffs);

  for (const auto& channel : *coeffs) {
    for (uint y = 0; y < frame->rows; y += block_h) {
      for (uint x = 0; x < frame->cols; x += block_w) {
        cv::Rect roi(x, y, block_w, block_h);
        cv::Mat1f block = channel(roi);
        cv::dct(block, block);
      }
    }
  }
}

void Encoder::operator()() {
  attempted_first_frame_read_.wait();

  if (in_queue_.IsEmpty()) {
    // An empty queue after attempting to read the first frame implies the
    // reader is done.
    return;
  }

  cv::Mat3b frame;
  bool empty_and_reader_done = !in_queue_.Pop(frame);

  std::vector<cv::Mat1f> dct_coeffs(frame.channels());
  for (auto& ch : dct_coeffs) {
    ch = cv::Mat1f(padded_frame_h_, padded_frame_w_);
  }

  cv::Mat3f padded_frame_float(padded_frame_h_, padded_frame_w_);

  {
    // TODO: Support I-frames
    // First frame is not included in the encoded video. It is used as a
    // tracked frame and not an anchor frame.
    uint frame_count = vidprops_.frame_count;
    if (frame_count > 0) {
      --frame_count;
    }
    Header h{frame_count,
             vidprops_.frame_w,
             vidprops_.frame_h,
             frame_excess_w_,
             frame_excess_h_,
             cfg_.transform_block_w,
             cfg_.transform_block_h,
             static_cast<uint>(frame.channels())};

    auto first = reinterpret_cast<uchar*>(&h);
    auto last = first + sizeof(h);
    std::vector<uchar> buf(first, last);
    out_queue_.Push(std::move(buf));
  }

#ifdef VISUALIZE
  //---------------------------------------------------------------------------
  // ALLOCATE WINDOW VIEWS AND OTHER VISUALIZATION-RELATED DATA STRUCTURES
  //---------------------------------------------------------------------------
  cv::Mat3b window_views;
  // window row 1
  cv::Mat3b pyr_base_view;
  cv::Mat3b motion_field_view;
  cv::Mat3b global_motion_view;
  // window row 2
  cv::Mat3b foreground_mask_view;
  cv::Mat3b foreground_mask_after_morph_view;
  cv::Mat3b foreground_clusters_view;
  // window row 3
  cv::Mat3b foreground_regions_view;
  {
    int w = padded_frame_w_;
    int h = padded_frame_h_;

    window_views = cv::Mat3b(h * 3, w * 3);

    cv::Rect r(cv::Point2i(0, 0), cv::Size2i(w, h));
    pyr_base_view = window_views(r);
    motion_field_view = window_views(r + cv::Point2i(w, 0));
    global_motion_view = window_views(r + cv::Point2i(w * 2, 0));

    r.y = h;
    foreground_mask_view = window_views(r);
    foreground_mask_after_morph_view = window_views(r + cv::Point2i(w, 0));
    foreground_clusters_view = window_views(r + cv::Point2i(w * 2, 0));

    r.y = 2 * h;
    foreground_regions_view = window_views(r);
  }

  cv::Mat1b foreground_mask_frame(padded_frame_h_, padded_frame_w_);
  cv::Mat3b foreground_cluster_frame(mv_field_h_, mv_field_w_);
  cv::Mat3b foreground_regions_frame(mv_field_h_, mv_field_w_);

  std::vector<uint> foreground_cluster_ids(mv_field_.size());

  cv::namedWindow(kWindowName);

  ArrowedLineParams line_params;
  DefaultInit(&line_params);

  ViewTitleTextParams vtt_params;
  vtt_params.outline_color = cv::Scalar(0, 0, 0);
  vtt_params.fill_color = cv::Scalar(255, 255, 255);
  vtt_params.font = cv::HersheyFonts::FONT_HERSHEY_COMPLEX;
  vtt_params.font_scale_factor =
      (1.0f / 640.0f) * Min(padded_frame_w_, padded_frame_h_);
  {
    float text_origin_scale_factor = 2 * vtt_params.font_scale_factor;
    vtt_params.origin = cv::Point2i(8, 16);
    vtt_params.origin.x =
        RoundFloatToInt(vtt_params.origin.x * text_origin_scale_factor);
    vtt_params.origin.y =
        RoundFloatToInt(vtt_params.origin.y * text_origin_scale_factor);
  }
  vtt_params.line_type = cv::LineTypes::LINE_AA;
  vtt_params.line_thickness_scale_factor = vtt_params.font_scale_factor;
#endif  // VISUALIZE

  cv::copyMakeBorder(frame, padded_frame_, 0, frame_excess_h_, 0,
                     frame_excess_w_, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::cvtColor(padded_frame_, yuv_padded_frame_, cv::COLOR_BGR2YUV);
  cv::extractChannel(yuv_padded_frame_, prev_y_padded_frame_, 0);
  cv::buildPyramid(prev_y_padded_frame_, prev_pyr_, cfg_.pyr_lvl_count - 1);

  // while (!shared_reader_data_.reader_is_done || !inqueue.IsEmpty()) {
  while (true) {
    bool empty_and_reader_done = !in_queue_.Pop(frame);
    if (empty_and_reader_done) {
      break;
    }

    cv::copyMakeBorder(frame, padded_frame_, 0, frame_excess_h_, 0,
                       frame_excess_w_, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

#ifdef VISUALIZE
    padded_frame_.copyTo(pyr_base_view);
    DrawViewTitle(&pyr_base_view, "Base", &vtt_params);
#endif  // VISUALIZE

    cv::cvtColor(padded_frame_, yuv_padded_frame_, cv::COLOR_BGR2YUV);
    cv::extractChannel(yuv_padded_frame_, y_padded_frame_, 0);
    cv::buildPyramid(y_padded_frame_, pyr_, cfg_.pyr_lvl_count - 1);

#if defined(__SSE2__) && defined(SVC_MOTION_SSE2)
    EstimateMotionHierarchical16x16Sse2(prev_pyr_data_.data(), pyr_data_.data(),
                                        padded_frame_w_, padded_frame_h_,
                                        cfg_.mv_search_range, mv_field_.data(),
                                        mv_field_min_mad_.data());
#else
    EstimateMotionHierarchical(
        prev_pyr_data_.data(), pyr_data_.data(), cfg_.pyr_lvl_count,
        padded_frame_w_, padded_frame_h_, cfg_.mv_search_range, cfg_.mv_block_w,
        cfg_.mv_block_h, mv_field_.data(), mv_field_min_mad_.data());
#endif

#ifdef VISUALIZE
    padded_frame_.copyTo(motion_field_view);
    DrawMotionField(mv_field_.data(), cfg_.mv_block_w, cfg_.mv_block_h,
                    &line_params, &motion_field_view);
    DrawViewTitle(&motion_field_view, "Motion Field (MF)", &vtt_params);
#endif  // VISUALIZE

    Vec2f global_motion;
    std::vector<uint> background_mv_field_indices;
    {
      float rmse;
      EstimateGlobalMotionRansac(mv_field_.data(), mv_field_.size(),
                                 cfg_.ransac, &rmse, &global_motion,
                                 &background_mv_field_indices);
    }

#ifdef VISUALIZE
    padded_frame_.copyTo(global_motion_view);
    DrawMotionVecAsField(global_motion, cfg_.mv_block_w, cfg_.mv_block_h,
                         &line_params, &global_motion_view);
    DrawViewTitle(&global_motion_view, "Global Motion (GM)", &vtt_params);
#endif  // VISUALIZE

    foreground_mv_field_mask_ = cv::Mat1b::ones(mv_field_h_, mv_field_w_) * 255;
    {
      uchar* fg_mask = foreground_mv_field_mask_.ptr<uchar>();
      for (uint i : background_mv_field_indices) {
        fg_mask[i] = 0;
      }
    }

#ifdef VISUALIZE
    cv::resize(foreground_mv_field_mask_, foreground_mask_frame,
               foreground_mask_frame.size(), 0, 0, cv::INTER_NEAREST_EXACT);
    cv::cvtColor(foreground_mask_frame, foreground_mask_view,
                 cv::COLOR_GRAY2BGR);
    DrawViewTitle(&foreground_mask_view, "Foreground (FG) Mask", &vtt_params);
#endif  // VISUALIZE

    // improve spatial connectivity of foreground mask
    cv::morphologyEx(foreground_mv_field_mask_, foreground_mv_field_mask_,
                     cv::MORPH_CLOSE, morph_rect_);
    cv::morphologyEx(foreground_mv_field_mask_, foreground_mv_field_mask_,
                     cv::MORPH_OPEN, morph_rect_);

#ifdef VISUALIZE
    cv::resize(foreground_mv_field_mask_, foreground_mask_frame,
               foreground_mask_frame.size(), 0, 0, cv::INTER_NEAREST_EXACT);
    cv::cvtColor(foreground_mask_frame, foreground_mask_after_morph_view,
                 cv::COLOR_GRAY2BGR);
    DrawViewTitle(&foreground_mask_after_morph_view, "FG Mask After Morph",
                  &vtt_params);
#endif  // VISUALIZE

    foreground_mv_field_indices_.clear();

    {
      uchar* fg_mask = foreground_mv_field_mask_.ptr<uchar>();
      for (uint i = 0; i < mv_field_.size(); ++i) {
        if (fg_mask[i] == 255) {
          foreground_mv_field_indices_.push_back(i);
        }
      }
    }

    for (uint& t : mv_field_block_types_) {
      t = BLOCK_TYPE_BACKGROUND;
    }

#ifdef VISUALIZE
    padded_frame_.copyTo(foreground_clusters_view);
#endif  // VISUALIZE

    if (!foreground_mv_field_indices_.empty()) {
      uint cluster_count =
          Min(cfg_.kmeans.cluster_count, foreground_mv_field_indices_.size());

      cv::Mat1i cluster_ids;
      {
        foreground_mv_features_.resize(foreground_mv_field_indices_.size());
        BuildMvFeatures(mv_field_.data(), mv_field_w_, cfg_.mv_block_w,
                        cfg_.mv_block_h, foreground_mv_field_indices_.data(),
                        foreground_mv_field_indices_.size(),
                        foreground_mv_features_.data());
        cv::Mat4f fg_mv_features(foreground_mv_features_.size(), 1,
                                 (cv::Vec4f*)foreground_mv_features_.data());

        cv::TermCriteria term_crit(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            cfg_.kmeans.max_iter_count, cfg_.kmeans.epsilon);

        cv::kmeans(fg_mv_features, cluster_count, cluster_ids, term_crit,
                   cfg_.kmeans.attempt_count, cv::KMEANS_PP_CENTERS);
      }

      int* cluster_ids_ = (int*)cluster_ids.data;

#ifdef VISUALIZE
      for (uint& cid : foreground_cluster_ids) {
        cid = 0;
      }
      for (uint i = 0; i < foreground_mv_field_indices_.size(); ++i) {
        uint j = foreground_mv_field_indices_[i];
        foreground_cluster_ids[j] = cluster_ids_[i] + BLOCK_TYPE_BACKGROUND + 1;
      }
      DrawVecFieldLayerClusters(
          foreground_cluster_ids.data(), BLOCK_TYPE_BACKGROUND + 1,
          foreground_mv_field_indices_.data(),
          foreground_mv_field_indices_.size(), mv_field_w_, mv_field_h_,
          cfg_.mv_block_w, cfg_.mv_block_h, &foreground_clusters_view);
#endif  // VISUALIZE

      uint block_type_offset = BLOCK_TYPE_BACKGROUND;
      for (uint cid = 0; cid < cluster_count; ++cid) {
        foreground_cluster_mask_ = cv::Mat1b::zeros(mv_field_h_, mv_field_w_);

        for (uint i = 0; i < foreground_mv_field_indices_.size(); ++i) {
          if (cluster_ids_[i] == cid) {
            uint j = foreground_mv_field_indices_[i];
            foreground_cluster_mask_.data[j] = 255;
          }
        }

        cv::Mat1i conn_comp_ids;
        uint conn_comp_id_count = cv::connectedComponents(
            foreground_cluster_mask_, conn_comp_ids,
            cfg_.connected_components_connectivity, CV_32S,
            cv::ConnectedComponentsAlgorithmsTypes::CCL_DEFAULT);

        int* conn_comp_ids_ = (int*)conn_comp_ids.data;
        for (uint i : foreground_mv_field_indices_) {
          if (conn_comp_ids_[i] == 0) {
            continue;
          }
          mv_field_block_types_[i] = conn_comp_ids_[i] + block_type_offset;
        }

        block_type_offset += conn_comp_id_count;
      }
    }

#ifdef VISUALIZE
    DrawViewTitle(&foreground_clusters_view, "FG Clusters", &vtt_params);

    padded_frame_.copyTo(foreground_regions_view);
    DrawVecFieldLayerClusters(
        mv_field_block_types_.data(), BLOCK_TYPE_BACKGROUND + 1,
        foreground_mv_field_indices_.data(),
        foreground_mv_field_indices_.size(), mv_field_w_, mv_field_h_,
        cfg_.mv_block_w, cfg_.mv_block_h, &foreground_regions_view);
    DrawViewTitle(&foreground_regions_view, "FG Regions", &vtt_params);
#endif  // VISUALIZE

    padded_frame_.convertTo(padded_frame_float, CV_32FC3);
    Dct(&padded_frame_float, cfg_.transform_block_w, cfg_.transform_block_h,
        &dct_coeffs);

    std::vector<cv::Mat1f> dct_coeffs_copy(dct_coeffs.size());
    for (decltype(dct_coeffs)::size_type i = 0; i < dct_coeffs.size(); ++i) {
      dct_coeffs_copy[i] = dct_coeffs[i].clone();
    }

    std::vector<uchar> bytes = SerializeEncodedFrame(
        &dct_coeffs_copy, &mv_field_block_types_, vidprops_.frame_w,
        vidprops_.frame_h, cfg_.transform_block_w, cfg_.transform_block_h,
        mv_field_w_, mv_field_h_, cfg_.mv_block_w, cfg_.mv_block_h);

    out_queue_.Push(std::move(bytes));

#ifdef VISUALIZE
    cv::imshow(kWindowName, window_views);
    if (cv::waitKey(1) >= 0) {
      break;
    }
#endif  // VISUALIZE

    prev_pyr_.swap(pyr_);
    prev_pyr_data_.swap(pyr_data_);
    cv::swap(prev_y_padded_frame_, y_padded_frame_);
  }

  out_queue_.SignalProducerIsDone();

#ifdef VISUALIZE
  cv::destroyAllWindows();
#endif  // VISUALIZE
}