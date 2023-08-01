#include "decoder.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "codec.hpp"
#include "error.hpp"
#include "queue.hpp"
#include "thread.hpp"
#include "types.hpp"

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

static void DefaultInit(DecoderConfig& c) {
  c.foreground_quant_step = 1;
  c.background_quant_step = 640;
  c.max_gaze_rect_w = 64;
  c.max_gaze_rect_h = 64;
}

static Error ParseConfig(uint argc, char* argv[], DecoderConfig& c) {
  assert(argv);

  /*******************************************************************************
   * Command-line Options    #options
   *******************************************************************************/
  cli::Opt opts[]{
      {"foreground-quant-step", cli::kOptArgType_Uint,
       &c.foreground_quant_step},
      {"background-quant-step", cli::kOptArgType_Uint,
       &c.background_quant_step},
      {"max-gaze-rect-w", cli::kOptArgType_Uint, &c.max_gaze_rect_w},
      {"max-gaze-rect-h", cli::kOptArgType_Uint, &c.max_gaze_rect_h}};

  uint argi;
  uint opts_size = sizeof(opts) / sizeof(opts[0]);

  cli::Status status = cli::ParseOpts(argc, argv, opts, opts_size, &argi);
  if (status != cli::kStatus_Ok) {
    std::string msg{"parsing options: "};
    msg += cli::StatusMessage(status);
    return Error{ErrorCode::kUnspecified, std::move(msg)};
  }

  return Error{ErrorCode::kOk};
}

struct Reader {
  Header header;
  CircularQueue<std::vector<std::byte>>& queue;

  void operator()() {
    auto blk_area_per_channel =
        header.transform_block_w * header.transform_block_h;
    auto blk_area = blk_area_per_channel * header.channel_count;
    auto blk_type_sz = sizeof(uint);
    auto byte_count = blk_type_sz + sizeof(float) * blk_area;

    uint upscaled_frame_w = header.frame_w + header.frame_excess_w;
    uint upscaled_frame_h = header.frame_h + header.frame_excess_h;

    for (uint i = 0; i < header.frame_count; ++i) {
      for (uint y = 0; y < upscaled_frame_h; y += header.transform_block_h) {
        for (uint x = 0; x < upscaled_frame_w; x += header.transform_block_w) {
          std::vector<std::byte> block(byte_count);

          auto count = std::fread(block.data(), 1, byte_count, stdin);
          if (count < byte_count) {
            throw std::runtime_error{"failed to read block"};
          }

          queue.Push(std::move(block));
        }
      }
    }

    queue.SignalProducerIsDone();
  }
};

static CircularQueue<std::vector<std::byte>> in_blocks{100};

int main(int argc, char* argv[]) {
  DecoderConfig cfg;
  DefaultInit(cfg);

  auto err = ParseConfig(argc, argv, cfg);
  if (err.code != ErrorCode::kOk) {
    std::fprintf(stderr, "parsing config: %s\n", err.message.c_str());
    return EXIT_FAILURE;
  }

  err = Validate(cfg);
  if (err.code != ErrorCode::kOk) {
    std::fprintf(stderr, "validating config: %s\n", err.message.c_str());
    return EXIT_FAILURE;
  }

  Header header;
  auto count = std::fread(&header, sizeof(header), 1, stdin);
  if (count == 0) {
    std::fprintf(stderr, "failed to read header\n");
    return EXIT_FAILURE;
  }

  Reader reader{header, in_blocks};
  Decoder decoder{cfg, header, in_blocks};

  std::thread read_thread{reader};
  ThreadGuard rtg{read_thread};

  decoder();
}