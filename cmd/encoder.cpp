#include <cassert>
#include <opencv2/core.hpp>
#ifdef VISUALIZE
#include <opencv2/highgui.hpp>
#endif  // VISUALIZE
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cli.hpp"
#include "codec.hpp"
#include "encoder.hpp"
#include "motion.hpp"
#include "types.hpp"
#ifdef VISUALIZE
#include "draw.hpp"
#endif  // VISUALIZE

struct Config {
  char* video_path;
  boolean verbose;
  EncoderConfig encoder;
};

void DefaultInit(Config* c) {
  assert(c);

  DefaultInit(&c->encoder);

  c->video_path = 0;
  c->verbose = 1;
}

static Status ParseConfig(uint argc, char* argv[], Config* c) {
  assert(argv);
  assert(c);

  EncoderConfig* ec = &c->encoder;

  Status res = kStatus_InvalidParamError;

  /*******************************************************************************
   * Command-line Options    #options
   *******************************************************************************/
  cli::Opt opts[] {
#if !defined(__SSE2__) || !defined(SVC_MOTION_SSE2)
    {"mv-block-w", cli::kOptArgType_Uint, &ec->mv_block_w},
        {"mv-block-h", cli::kOptArgType_Uint, &ec->mv_block_h},
        {"pyr-lvl-count", cli::kOptArgType_Uint, &ec->pyr_lvl_count},
#endif
        {"mv-search-range", cli::kOptArgType_Uint, &ec->mv_search_range},
        {"ransac-subset-sz", cli::kOptArgType_Uint, &ec->ransac.subset_sz},
        {"ransac-inlier-thresh", cli::kOptArgType_Float,
         &ec->ransac.inlier_thresh},
        {"ransac-success-prob", cli::kOptArgType_Float,
         &ec->ransac.success_prob},
        {"ransac-inlier-ratio", cli::kOptArgType_Float,
         &ec->ransac.inlier_ratio},
        {"morph-rect-w", cli::kOptArgType_Uint, &ec->morph_rect_w},
        {"morph-rect-h", cli::kOptArgType_Uint, &ec->morph_rect_h},
        {"kmeans-cluster-count", cli::kOptArgType_Uint,
         &ec->kmeans.cluster_count},
        {"kmeans-attempt-count", cli::kOptArgType_Uint,
         &ec->kmeans.attempt_count},
        {"kmeans-max-iter-count", cli::kOptArgType_Uint,
         &ec->kmeans.max_iter_count},
        {"kmeans-epsilon", cli::kOptArgType_Float, &ec->kmeans.epsilon},
        {"connected-components-connectivity", cli::kOptArgType_Uint,
         &ec->connected_components_connectivity},
        {"transform-block-w", cli::kOptArgType_Uint, &ec->transform_block_w},
        {"transform-block-h", cli::kOptArgType_Uint, &ec->transform_block_h}, {
      "verbose", cli::kOptArgType_Int, &c->verbose
    }
  };

  uint argi;
  uint opts_size = sizeof(opts) / sizeof(opts[0]);

  cli::Status st = cli::ParseOpts(argc, argv, opts, opts_size, &argi);
  if (st != cli::kStatus_Ok) {
    std::fprintf(stderr, "parsing options: %s\n", cli::StatusMessage(st));
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

static void BuildMvFeatures(const Vec2f* mv_field, uint mv_field_w,
                            uint mv_block_w, uint mv_block_h,
                            const uint* mv_field_indices,
                            uint mv_field_indices_sz, Vec4f* features) {
  assert(mv_field);
  assert(features);

  assert(mv_field_w > 0);

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

#ifndef VISUALIZE
static void InitHeader(Header* h, Encoder* e, uint frame_count,
                       uint channel_count) {
  assert(h);
  assert(e);

  h->frame_count = frame_count;
  h->frame_w = e->cfg.frame_w;
  h->frame_h = e->cfg.frame_h;
  h->frame_excess_w = e->padded_frame_w - e->cfg.frame_w;
  h->frame_excess_h = e->padded_frame_h - e->cfg.frame_h;
  h->transform_block_w = e->cfg.transform_block_w;
  h->transform_block_h = e->cfg.transform_block_h;
  h->channel_count = channel_count;
}

static void Dct(const cv::Mat3f* frame, uint block_w, uint block_h,
                std::vector<cv::Mat1f>* coeffs) {
  assert(frame);
  assert(coeffs);

  assert(block_h > 0);
  assert(block_w > 0);

  cv::split(*frame, *coeffs);

  for (const cv::Mat1f& channel : *coeffs) {
    for (uint y = 0; y < frame->rows; y += block_h) {
      for (uint x = 0; x < frame->cols; x += block_w) {
        cv::Rect roi(x, y, block_w, block_h);
        cv::Mat1f block = channel(roi);
        cv::dct(block, block);
      }
    }
  }
}

static Status WriteEncodedFrame(const std::vector<cv::Mat1f>* dct_coeffs,
                                const std::vector<uint>* mv_field_block_types,
                                uint frame_w, uint frame_h,
                                uint transform_block_w, uint transform_block_h,
                                uint mv_field_w, uint mv_field_h,
                                uint mv_block_w, uint mv_block_h) {
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

  Status res = kStatus_IoError;

  for (uint tb_y = 0; tb_y < frame_h; tb_y += transform_block_w) {
    for (uint tb_x = 0; tb_x < frame_w; tb_x += transform_block_h) {
      uint mv_field_y = tb_y / mv_block_h;
      uint mv_field_x = tb_x / mv_block_w;
      uint mv_field_i = mv_field_y * mv_field_w + mv_field_x;

      uint btype = (*mv_field_block_types)[mv_field_i];

      uint count = std::fwrite(&btype, sizeof(btype), 1, stdout);
      if (count < 1) {
        std::fprintf(stderr, "failed to write block type\n");
        return res;
      }

      for (const cv::Mat1f& channel : (*dct_coeffs)) {
        float* coeffs = (float*)channel.data;

        for (uint y = tb_y; y < tb_y + transform_block_w; ++y) {
          float* row = &coeffs[y * frame_w + tb_x];

          uint count =
              std::fwrite(row, sizeof(float), transform_block_h, stdout);
          if (count < transform_block_h) {
            std::fprintf(stderr, "failed to write DCT coefficients\n");
            return res;
          }
        }
      }
    }
  }

  res = kStatus_Ok;

  return res;
}
#endif  // NOT VISUALIZE

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

int main(int argc, char* argv[]) {
  Config cfg;
  DefaultInit(&cfg);

  Status status = ParseConfig(argc, argv, &cfg);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to parse configuration\n");
    return EXIT_FAILURE;
  }

  cv::VideoCapture vidcap(cfg.video_path);

  if (!vidcap.isOpened()) {
    std::fprintf(stderr, "failed to initialize video capturing\n");
    return EXIT_FAILURE;
  }

  cfg.encoder.frame_w =
      vidcap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
  cfg.encoder.frame_h =
      vidcap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

  status = Validate(&cfg.encoder);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to validate configuration\n");
    return EXIT_FAILURE;
  }

  uint frame_count =
      vidcap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);

  if (cfg.verbose) {
    std::fprintf(stderr, "video properties:\n");
    std::fprintf(stderr, "  width: %u\n", cfg.encoder.frame_w);
    std::fprintf(stderr, "  height: %u\n", cfg.encoder.frame_h);
    std::fprintf(stderr, "  frame count: %u\n", frame_count);
  }

  cv::Mat3b frame;

  if (!vidcap.read(frame)) {
    std::fprintf(stderr, "no frames in video file\n");
    return EXIT_SUCCESS;
  }

  Encoder enc;
  Init(&enc, &cfg.encoder);

  uint frame_excess_w = enc.padded_frame_w - enc.cfg.frame_w;
  uint frame_excess_h = enc.padded_frame_h - enc.cfg.frame_h;

  cv::Mat1b foreground_mv_field_mask(enc.mv_field_h, enc.mv_field_w,
                                     enc.foreground_mv_field_mask.data());

  cv::Mat1b foreground_cluster_mask(enc.mv_field_h, enc.mv_field_w);

  cv::Mat morph_rect = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(enc.cfg.morph_rect_w, enc.cfg.morph_rect_h));

#ifndef VISUALIZE
  std::vector<cv::Mat1f> dct_coeffs(frame.channels());
  for (cv::Mat1f& channel : dct_coeffs) {
    channel = cv::Mat1f(enc.padded_frame_h, enc.padded_frame_w);
  }

  cv::Mat3f padded_frame_float(enc.padded_frame_h, enc.padded_frame_w);

  {
    Header header;
    // TODO: Support I-frames
    // First frame is not included in the encoded video. It is used as a
    // tracked frame and not an anchor frame.
    if (frame_count > 0) {
      --frame_count;
    }
    InitHeader(&header, &enc, frame_count, frame.channels());

    uint count = std::fwrite(&header, sizeof(header), 1, stdout);
    if (count < 1) {
      std::fprintf(stderr, "failed to write header\n");
      return EXIT_FAILURE;
    }
  }
#endif  // NOT VISUALIZE

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
    int w = enc.padded_frame_w;
    int h = enc.padded_frame_h;

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

  cv::Mat1b foreground_mask_frame(enc.padded_frame_h, enc.padded_frame_w);
  cv::Mat3b foreground_cluster_frame(enc.mv_field_h, enc.mv_field_w);
  cv::Mat3b foreground_regions_frame(enc.mv_field_h, enc.mv_field_w);

  std::vector<uint> foreground_cluster_ids(enc.mv_field.size());

  cv::namedWindow(kWindowName);

  ArrowedLineParams line_params;
  DefaultInit(&line_params);

  ViewTitleTextParams vtt_params;
  vtt_params.outline_color = cv::Scalar(0, 0, 0);
  vtt_params.fill_color = cv::Scalar(255, 255, 255);
  vtt_params.font = cv::HersheyFonts::FONT_HERSHEY_COMPLEX;
  vtt_params.font_scale_factor =
      (1.0f / 640.0f) * Min(enc.padded_frame_w, enc.padded_frame_h);
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

  cv::copyMakeBorder(frame, enc.padded_frame, 0, frame_excess_h, 0,
                     frame_excess_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::cvtColor(enc.padded_frame, enc.yuv_padded_frame, cv::COLOR_BGR2YUV);
  cv::extractChannel(enc.yuv_padded_frame, enc.prev_y_padded_frame, 0);
  cv::buildPyramid(enc.prev_y_padded_frame, enc.prev_pyr,
                   enc.cfg.pyr_lvl_count - 1);

  while (vidcap.read(frame)) {
    cv::copyMakeBorder(frame, enc.padded_frame, 0, frame_excess_h, 0,
                       frame_excess_w, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

#ifdef VISUALIZE
    enc.padded_frame.copyTo(pyr_base_view);
    DrawViewTitle(&pyr_base_view, "Base", &vtt_params);
#endif  // VISUALIZE

    cv::cvtColor(enc.padded_frame, enc.yuv_padded_frame, cv::COLOR_BGR2YUV);
    cv::extractChannel(enc.yuv_padded_frame, enc.y_padded_frame, 0);
    cv::buildPyramid(enc.y_padded_frame, enc.pyr, enc.cfg.pyr_lvl_count - 1);

#if defined(__SSE2__) && defined(SVC_MOTION_SSE2)
    EstimateMotionHierarchical16x16Sse2(
        enc.prev_pyr_data.data(), enc.pyr_data.data(), enc.padded_frame_w,
        enc.padded_frame_h, enc.cfg.mv_search_range, enc.mv_field.data(),
        enc.mv_field_min_mad.data());
#else
    EstimateMotionHierarchical(
        enc.prev_pyr_data.data(), enc.pyr_data.data(), enc.cfg.pyr_lvl_count,
        enc.padded_frame_w, enc.padded_frame_h, enc.cfg.mv_search_range,
        enc.cfg.mv_block_w, enc.cfg.mv_block_h, enc.mv_field.data(),
        enc.mv_field_min_mad.data());
#endif

#ifdef VISUALIZE
    enc.padded_frame.copyTo(motion_field_view);
    DrawMotionField(enc.mv_field.data(), enc.cfg.mv_block_w, enc.cfg.mv_block_h,
                    &line_params, &motion_field_view);
    DrawViewTitle(&motion_field_view, "Motion Field (MF)", &vtt_params);
#endif  // VISUALIZE

    Vec2f global_motion;
    std::vector<uint> background_mv_field_indices;
    {
      float rmse;
      EstimateGlobalMotionRansac(enc.mv_field.data(), enc.mv_field.size(),
                                 enc.cfg.ransac, &rmse, &global_motion,
                                 &background_mv_field_indices);
    }

#ifdef VISUALIZE
    enc.padded_frame.copyTo(global_motion_view);
    DrawMotionVecAsField(global_motion, enc.cfg.mv_block_w, enc.cfg.mv_block_h,
                         &line_params, &global_motion_view);
    DrawViewTitle(&global_motion_view, "Global Motion (GM)", &vtt_params);
#endif  // VISUALIZE

    foreground_mv_field_mask =
        cv::Mat1b::ones(enc.mv_field_h, enc.mv_field_w) * 255;
    for (uint i : background_mv_field_indices) {
      enc.foreground_mv_field_mask[i] = 0;
    }

#ifdef VISUALIZE
    cv::resize(foreground_mv_field_mask, foreground_mask_frame,
               foreground_mask_frame.size(), 0, 0, cv::INTER_NEAREST_EXACT);
    cv::cvtColor(foreground_mask_frame, foreground_mask_view,
                 cv::COLOR_GRAY2BGR);
    DrawViewTitle(&foreground_mask_view, "Foreground (FG) Mask", &vtt_params);
#endif  // VISUALIZE

    // improve spatial connectivity of foreground mask
    cv::morphologyEx(foreground_mv_field_mask, foreground_mv_field_mask,
                     cv::MORPH_CLOSE, morph_rect);
    cv::morphologyEx(foreground_mv_field_mask, foreground_mv_field_mask,
                     cv::MORPH_OPEN, morph_rect);

#ifdef VISUALIZE
    cv::resize(foreground_mv_field_mask, foreground_mask_frame,
               foreground_mask_frame.size(), 0, 0, cv::INTER_NEAREST_EXACT);
    cv::cvtColor(foreground_mask_frame, foreground_mask_after_morph_view,
                 cv::COLOR_GRAY2BGR);
    DrawViewTitle(&foreground_mask_after_morph_view, "FG Mask After Morph",
                  &vtt_params);
#endif  // VISUALIZE

    enc.foreground_mv_field_indices.clear();
    for (uint i = 0; i < enc.mv_field.size(); ++i) {
      if (enc.foreground_mv_field_mask[i] == 255) {
        enc.foreground_mv_field_indices.push_back(i);
      }
    }

    for (uint& t : enc.mv_field_block_types) {
      t = BLOCK_TYPE_BACKGROUND;
    }

#ifdef VISUALIZE
    enc.padded_frame.copyTo(foreground_clusters_view);
#endif  // VISUALIZE

    if (!enc.foreground_mv_field_indices.empty()) {
      uint cluster_count = Min(enc.cfg.kmeans.cluster_count,
                               enc.foreground_mv_field_indices.size());

      cv::Mat1i cluster_ids;
      {
        enc.foreground_mv_features.resize(
            enc.foreground_mv_field_indices.size());
        BuildMvFeatures(enc.mv_field.data(), enc.mv_field_w, enc.cfg.mv_block_w,
                        enc.cfg.mv_block_h,
                        enc.foreground_mv_field_indices.data(),
                        enc.foreground_mv_field_indices.size(),
                        enc.foreground_mv_features.data());
        cv::Mat4f fg_mv_features(enc.foreground_mv_features.size(), 1,
                                 (cv::Vec4f*)enc.foreground_mv_features.data());

        cv::TermCriteria term_crit(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            enc.cfg.kmeans.max_iter_count, enc.cfg.kmeans.epsilon);

        cv::kmeans(fg_mv_features, cluster_count, cluster_ids, term_crit,
                   enc.cfg.kmeans.attempt_count, cv::KMEANS_PP_CENTERS);
      }

      int* cluster_ids_ = (int*)cluster_ids.data;

#ifdef VISUALIZE
      for (uint& cid : foreground_cluster_ids) {
        cid = 0;
      }
      for (uint i = 0; i < enc.foreground_mv_field_indices.size(); ++i) {
        uint j = enc.foreground_mv_field_indices[i];
        foreground_cluster_ids[j] = cluster_ids_[i] + BLOCK_TYPE_BACKGROUND + 1;
      }
      DrawVecFieldLayerClusters(
          foreground_cluster_ids.data(), BLOCK_TYPE_BACKGROUND + 1,
          enc.foreground_mv_field_indices.data(),
          enc.foreground_mv_field_indices.size(), enc.mv_field_w,
          enc.mv_field_h, enc.cfg.mv_block_w, enc.cfg.mv_block_h,
          &foreground_clusters_view);
#endif  // VISUALIZE

      uint block_type_offset = BLOCK_TYPE_BACKGROUND;
      for (uint cid = 0; cid < cluster_count; ++cid) {
        foreground_cluster_mask =
            cv::Mat1b::zeros(enc.mv_field_h, enc.mv_field_w);

        for (uint i = 0; i < enc.foreground_mv_field_indices.size(); ++i) {
          if (cluster_ids_[i] == cid) {
            uint j = enc.foreground_mv_field_indices[i];
            foreground_cluster_mask.data[j] = 255;
          }
        }

        cv::Mat1i conn_comp_ids;
        uint conn_comp_id_count = cv::connectedComponents(
            foreground_cluster_mask, conn_comp_ids,
            enc.cfg.connected_components_connectivity, CV_32S,
            cv::ConnectedComponentsAlgorithmsTypes::CCL_DEFAULT);

        int* conn_comp_ids_ = (int*)conn_comp_ids.data;
        for (uint i : enc.foreground_mv_field_indices) {
          if (conn_comp_ids_[i] == 0) {
            continue;
          }
          enc.mv_field_block_types[i] = conn_comp_ids_[i] + block_type_offset;
        }

        block_type_offset += conn_comp_id_count;
      }
    }

#ifdef VISUALIZE
    DrawViewTitle(&foreground_clusters_view, "FG Clusters", &vtt_params);

    enc.padded_frame.copyTo(foreground_regions_view);
    DrawVecFieldLayerClusters(
        enc.mv_field_block_types.data(), BLOCK_TYPE_BACKGROUND + 1,
        enc.foreground_mv_field_indices.data(),
        enc.foreground_mv_field_indices.size(), enc.mv_field_w, enc.mv_field_h,
        enc.cfg.mv_block_w, enc.cfg.mv_block_h, &foreground_regions_view);
    DrawViewTitle(&foreground_regions_view, "FG Regions", &vtt_params);
#endif  // VISUALIZE

#ifndef VISUALIZE
    enc.padded_frame.convertTo(padded_frame_float, CV_32FC3);
    Dct(&padded_frame_float, enc.cfg.transform_block_w,
        enc.cfg.transform_block_h, &dct_coeffs);

    status = WriteEncodedFrame(
        &dct_coeffs, &enc.mv_field_block_types, enc.padded_frame_w,
        enc.padded_frame_h, enc.cfg.transform_block_w,
        enc.cfg.transform_block_h, enc.mv_field_w, enc.mv_field_h,
        enc.cfg.mv_block_w, enc.cfg.mv_block_h);
    if (status != kStatus_Ok) {
      std::fprintf(stderr, "failed to write encoded frame\n");
      return EXIT_FAILURE;
    }
#endif  // NOT VISUALIZE

#ifdef VISUALIZE
    cv::imshow(kWindowName, window_views);
    if (cv::waitKey(1) >= 0) {
      break;
    }
#endif  // VISUALIZE

    enc.prev_pyr.swap(enc.pyr);
    enc.prev_pyr_data.swap(enc.pyr_data);
    cv::swap(enc.prev_y_padded_frame, enc.y_padded_frame);
  }

#ifdef VISUALIZE
  cv::destroyAllWindows();
#endif  // VISUALIZE
}