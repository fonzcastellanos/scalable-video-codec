#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <shared_mutex>
#include <vector>

#include "codec.hpp"
#include "decoder_config.hpp"
#include "math.hpp"
#include "types.hpp"

const char* kWindowName = "Decoded Video";

struct Block {
  uint type;
  std::vector<cv::Mat1f> channels;
};

static Status ReadBlock(uint channel_count, uint block_w, uint block_h,
                             Block* block) {
  assert(block);

  uint count = std::fread(&block->type, sizeof(block->type), 1, stdin);
  if (count < 1) {
    std::fprintf(stderr, "failed to read block type\n");
    return kStatus_IoError;
  }

  block->channels.resize(channel_count);

  uint block_area = block_w * block_h;
  uint char_to_read_count = sizeof(float) * block_area;

  for (cv::Mat1f& ch : block->channels) {
    ch = cv::Mat1f(block_h, block_w);
    uint count = std::fread(ch.data, sizeof(float), block_area, stdin);
    if (count < block_area) {
      std::fprintf(stderr, "failed to read dct coefficients\n");
      return kStatus_IoError;
    }
  }

  return kStatus_Ok;
}

static Status DecodeBlock(uint num_channels, uint block_w, uint block_h,
                               boolean gazed, uint fg_quant_step,
                               uint bg_quant_step, cv::Mat3f* decoded_block) {
  assert(decoded_block);

  Block block;

  Status status = ReadBlock(num_channels, block_w, block_h, &block);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to read block\n");
    return kStatus_IoError;
  }

  uint quant_step = fg_quant_step;
  if (gazed) {
    quant_step = 1;
  } else if (block.type == BLOCK_TYPE_BACKGROUND) {
    quant_step = bg_quant_step;
  }

  for (cv::Mat1f& channel : block.channels) {
    float* ch = (float*)channel.data;
    for (uint i = 0; i < channel.total(); ++i) {
      ch[i] /= quant_step;
      ch[i] = std::round(ch[i]);
      ch[i] *= quant_step;
    }
    cv::idct(channel, channel);
  }

  cv::merge(block.channels, *decoded_block);

  return kStatus_Ok;
}

static Status DecodeFrame(uint channel_count, uint frame_w, uint frame_h,
                               uint block_w, uint block_h, uint fg_quant_step,
                               uint bg_quant_step, cv::Rect2i gaze_region,
                               cv::Mat3f* decoded_frame) {
  assert(decoded_frame);

  for (uint y = 0; y < frame_h; y += block_h) {
    for (uint x = 0; x < frame_w; x += block_w) {
      cv::Rect roi(x, y, block_w, block_h);
      cv::Mat3f decoded_block = (*decoded_frame)(roi);

      boolean gazed = gaze_region.contains(cv::Point2i(x, y));

      Status status =
          DecodeBlock(channel_count, block_w, block_h, gazed, fg_quant_step,
                      bg_quant_step, &decoded_block);
      if (status != kStatus_Ok) {
        std::fprintf(stderr, "failed to decode block\n");
        return kStatus_UnspecifiedError;
      }
    }
  }

  return kStatus_Ok;
}

static cv::Rect2i CalcWithinFrameRectFromCenter(cv::Point2i center,
                                                uint max_rect_w,
                                                uint max_rect_h, uint frame_w,
                                                uint frame_h) {
  // TODO: add more assertions
  assert(center.x >= 0);
  assert(center.x < frame_w);

  assert(center.y >= 0);
  assert(center.y < frame_h);

  uint half_w = (max_rect_w + 1) / 2;
  if (center.x + half_w >= frame_w) {
    half_w = frame_w - center.x - 1;
  }
  if (center.x < half_w) {
    half_w = center.x;
  }

  uint half_h = (max_rect_h + 1) / 2;
  if (center.y + half_h >= frame_h) {
    half_h = frame_h - center.y - 1;
  }
  if (center.y < half_h) {
    half_h = center.y;
  }

  cv::Point2i tl;
  tl.x = center.x - half_w;
  tl.y = center.y - half_h;

  uint br_x = center.x + half_w;
  uint br_y = center.y + half_h;

  cv::Size2i size;
  size.width = br_x - tl.x;
  size.height = br_y - tl.y;

  cv::Rect2i result(tl, size);
  return result;
}

struct SharedVec2 {
  int x;
  int y;
  std::shared_mutex mutex;
};

void OnMouse(int event, int x, int y, int flags, void* mouse_position) {
  switch (event) {
    case cv::EVENT_MOUSEMOVE: {
      SharedVec2& pos = *static_cast<SharedVec2*>(mouse_position);
      std::unique_lock lock(pos.mutex);
      pos.x = x;
      pos.y = y;
      break;
    }
  }
}

int main(int argc, char* argv[]) {
  Config cfg;
  DefaultInit(&cfg);

  Status status = ParseConfig(argc, argv, &cfg);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to parse config\n");
    return EXIT_FAILURE;
  }

  status = Validate(&cfg);
  if (status != kStatus_Ok) {
    std::fprintf(stderr, "failed to validate config\n");
    return EXIT_FAILURE;
  }

  Header header;

  uint count = std::fread(&header, sizeof(header), 1, stdin);
  if (count == 0) {
    std::fprintf(stderr, "failed to read header\n");
    return EXIT_FAILURE;
  }

  cv::namedWindow(kWindowName);

  SharedVec2 mouse_pos = {};
  cv::setMouseCallback(kWindowName, OnMouse, &mouse_pos);

  uint upscaled_frame_w = header.frame_w + header.frame_excess_w;
  uint upscaled_frame_h = header.frame_h + header.frame_excess_h;

  cv::Mat3f upscaled_frame(upscaled_frame_h, upscaled_frame_w);
  cv::Mat3f frame(header.frame_h, header.frame_w);

  float w_ratio = (float)upscaled_frame_w / header.frame_w;
  float h_ratio = (float)upscaled_frame_h / header.frame_h;

  for (uint i = 0; i < header.frame_count; ++i) {
    cv::Point2i gaze_pos;
    {
      std::shared_lock lock(mouse_pos.mutex);
      gaze_pos.x = mouse_pos.x;
      gaze_pos.y = mouse_pos.y;
    }

    // gaze rectangle is defined in the original frame's space
    cv::Rect2i gaze_rect = CalcWithinFrameRectFromCenter(
        gaze_pos, cfg.max_gaze_rect_w, cfg.max_gaze_rect_h, header.frame_w,
        header.frame_h);

    // transform gaze rectangle to the upscaled frame's space
    gaze_rect.x = RoundFloatToInt(gaze_rect.x * w_ratio);
    gaze_rect.y = RoundFloatToInt(gaze_rect.y * h_ratio);
    gaze_rect.width = RoundFloatToInt(gaze_rect.width * w_ratio);
    gaze_rect.height = RoundFloatToInt(gaze_rect.height * h_ratio);

    status = DecodeFrame(header.channel_count, upscaled_frame_w,
                         upscaled_frame_h, header.transform_block_w,
                         header.transform_block_h, cfg.foreground_quant_step,
                         cfg.background_quant_step, gaze_rect, &upscaled_frame);
    if (status != kStatus_Ok) {
      std::fprintf(stderr, "failed to decode frame\n");
      return EXIT_FAILURE;
    }

    upscaled_frame /= 255;

    cv::resize(upscaled_frame, frame, frame.size(), 0, 0, CV_INTER_LINEAR);
    cv::imshow(kWindowName, frame);

    if (cv::waitKey(1) >= 0) {
      break;
    }
  }

  cv::destroyAllWindows();
}