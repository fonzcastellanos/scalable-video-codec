#include "encoder.hpp"

#include <cassert>
#include <cstdio>
#include <future>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "codec.hpp"
#include "motion.hpp"
#include "queue.hpp"
#include "thread.hpp"
#include "types.hpp"

struct Config {
  char* video_path;
  boolean verbose;
  EncoderConfig encoder;
};

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

static void DefaultInit(KMeansParams& p) {
  p.cluster_count = 10;
  p.attempt_count = 3;
  p.max_iter_count = 10;
  p.epsilon = 1;
}

static void DefaultInit(RansacParams& p) {
  p.subset_sz = 1;
  p.inlier_ratio = 0.5;
  p.success_prob = 0.99;
  p.inlier_thresh = 7.5;
}

static void DefaultInit(EncoderConfig& c) {
  c.mv_block_w = 16;
  c.mv_block_h = 16;
  c.mv_search_range = 8;
  c.pyr_lvl_count = 4;

  DefaultInit(c.ransac);

  c.morph_rect_w = 3;
  c.morph_rect_h = 3;

  DefaultInit(c.kmeans);

  c.connected_components_connectivity = 4;
  c.transform_block_w = 8;
  c.transform_block_h = 8;
}

static void DefaultInit(Config& c) {
  DefaultInit(c.encoder);

  c.video_path = 0;
  c.verbose = 1;
}

static Status ParseConfig(uint argc, char* argv[], Config& c) {
  assert(argv);

  EncoderConfig* ec = &c.encoder;

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
      "verbose", cli::kOptArgType_Int, &c.verbose
    }
  };

  uint argi;
  uint opts_size = sizeof(opts) / sizeof(opts[0]);

  cli::Status status = cli::ParseOpts(argc, argv, opts, opts_size, &argi);
  if (status != cli::kStatus_Ok) {
    std::fprintf(stderr, "Failed to parse options: %s.\n",
                 cli::StatusMessage(status));
    return kStatus_InvalidParamError;
  }

  if (argc < argi + 1) {
    std::fprintf(stderr, "Missing video path argument.\n");
    return kStatus_InvalidParamError;
  }

  c.video_path = argv[argi];

  return kStatus_Ok;
}

struct Reader {
  cv::VideoCapture& vc;
  CircularQueue<cv::Mat3b>& q;
  std::promise<void> attempted_first_frame_read;

  void operator()() {
    cv::Mat3b frame;

    if (!vc.read(frame)) {
      q.SignalProducerIsDone();
      attempted_first_frame_read.set_value();
      return;
    }

    q.Push(frame);

    attempted_first_frame_read.set_value();

    while (vc.read(frame)) {
      q.Push(frame);
    }

    q.SignalProducerIsDone();
  }
};

struct Writer {
  CircularQueue<std::vector<uchar>>& q;

  void operator()() {
    while (true) {
      std::vector<uchar> bytes;

      bool encoder_done = !q.Pop(bytes);
      if (encoder_done) {
        return;
      }

      uint count = std::fwrite(bytes.data(), 1, bytes.size(), stdout);
      if (count < bytes.size()) {
        std::fprintf(stderr, "Failed to write bytes.\n");
        return;
      }
    }
  }
};

static CircularQueue<cv::Mat3b> in_queue{10};
static CircularQueue<std::vector<uchar>> out_queue{10};

int main(int argc, char* argv[]) {
  Config cfg;
  DefaultInit(cfg);

  Status status = ParseConfig(argc, argv, cfg);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "Failed to parse configuration.\n");
    return EXIT_FAILURE;
  }

  cv::VideoCapture vidcap(cfg.video_path);
  if (!vidcap.isOpened()) {
    std::fprintf(stderr, "Failed to initialize video capturing.\n");
    return EXIT_FAILURE;
  }

  status = Validate(&cfg.encoder);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "Failed to validate configuration.\n");
    return EXIT_FAILURE;
  }

  VideoProperties vidprops{
      static_cast<uint>(
          vidcap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH)),
      static_cast<uint>(
          vidcap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT)),
      static_cast<uint>(
          vidcap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT))};

  if (cfg.verbose) {
    std::fprintf(stderr, "Video properties:\n");
    std::fprintf(stderr, "  Width: %u\n", vidprops.frame_w);
    std::fprintf(stderr, "  Height: %u\n", vidprops.frame_h);
    std::fprintf(stderr, "  Frame count: %u\n", vidprops.frame_count);
  }

  std::promise<void> attempted_first_frame_read_promise;
  auto attempted_first_frame_read_future =
      attempted_first_frame_read_promise.get_future();

  Reader reader{vidcap, in_queue,
                std::move(attempted_first_frame_read_promise)};
  Encoder encoder{cfg.encoder, vidprops, in_queue,
                  std::move(attempted_first_frame_read_future), out_queue};
  Writer writer{out_queue};

  std::thread read_thread{std::move(reader)};
  std::thread write_thread{std::move(writer)};
  ThreadGuard rtg{read_thread};
  ThreadGuard wtg{write_thread};

  encoder();
}