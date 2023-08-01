#include "decoder.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <shared_mutex>

#include "cli.hpp"
#include "math.hpp"

const char* kWindowName = "Decoded Video";

struct Block {
  uint type;
  std::vector<cv::Mat1f> channels;
};

struct SharedVec2 {
  int x;
  int y;
  std::shared_mutex mutex;
};

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

Error Validate(DecoderConfig& c) {
  if (c.foreground_quant_step == 0) {
    return Error{ErrorCode::kInvalidParameter,
                 "invalid foreground quantization step: must be > 0"};
  }

  if (c.background_quant_step == 0) {
    return Error{ErrorCode::kInvalidParameter,
                 "invalid background quantization step: must be > 0"};
  }

  return Error{ErrorCode::kOk};
}

Decoder::Decoder(const DecoderConfig& cfg, const Header& header,
                 CircularQueue<std::vector<std::byte>>& in_blocks)
    : cfg_{cfg}, header_{header}, in_blocks_{in_blocks} {}

static void OnMouse(int event, int x, int y, int flags, void* mouse_position) {
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

static void ParseBlock(const std::vector<std::byte>& raw_block,
                       uint channel_count, uint block_w, uint block_h,
                       Block& block) {
#ifndef NDEBUG
  uint blk_area_per_channel = block_w * block_h;
  uint blk_area = blk_area_per_channel * channel_count;
  uint size = sizeof(uint) + sizeof(float) * blk_area;
  assert(raw_block.size() == size);
#endif
  const uint* ptr = reinterpret_cast<const uint*>(raw_block.data());
  block.type = *ptr;

  const float* src_ptr = reinterpret_cast<const float*>(ptr + 1);

  block.channels.resize(channel_count);
  for (auto& ch : block.channels) {
    ch = cv::Mat1f(block_h, block_w);

    auto dst_ptr = ch.ptr<float>();
    std::memcpy(dst_ptr, src_ptr, sizeof(float) * ch.total());

    src_ptr += ch.total();
  }
}

static void DecodeBlock(Block& block, bool gazed, uint fg_quant_step,
                        uint bg_quant_step, cv::Mat3f& decoded_block) {
  uint quant_step = fg_quant_step;
  if (gazed) {
    quant_step = 1;
  } else if (block.type == BLOCK_TYPE_BACKGROUND) {
    quant_step = bg_quant_step;
  }

  for (auto& ch : block.channels) {
    auto ch_ptr = ch.ptr<float>();
    for (uint i = 0; i < ch.total(); ++i) {
      ch_ptr[i] /= quant_step;
      ch_ptr[i] = std::round(ch_ptr[i]);
      ch_ptr[i] *= quant_step;
    }
    cv::idct(ch, ch);
  }

  cv::merge(block.channels, decoded_block);
}

void Decoder::operator()() {
  cv::namedWindow(kWindowName);

  SharedVec2 mouse_pos = {};
  cv::setMouseCallback(kWindowName, OnMouse, &mouse_pos);

  uint upscaled_frame_w = header_.frame_w + header_.frame_excess_w;
  uint upscaled_frame_h = header_.frame_h + header_.frame_excess_h;

  cv::Mat3f upscaled_frame(upscaled_frame_h, upscaled_frame_w);
  cv::Mat3f frame(header_.frame_h, header_.frame_w);

  auto w_ratio = static_cast<float>(upscaled_frame_w) / header_.frame_w;
  auto h_ratio = static_cast<float>(upscaled_frame_h) / header_.frame_h;

  for (uint i = 0; i < header_.frame_count; ++i) {
    cv::Point2i gaze_pos;
    {
      std::shared_lock l{mouse_pos.mutex};
      gaze_pos.x = mouse_pos.x;
      gaze_pos.y = mouse_pos.y;
    }

    // gaze rectangle is defined in the original frame's space
    cv::Rect2i gaze_rect = CalcWithinFrameRectFromCenter(
        gaze_pos, cfg_.max_gaze_rect_w, cfg_.max_gaze_rect_h, header_.frame_w,
        header_.frame_h);

    // transform gaze rectangle to the upscaled frame's space
    gaze_rect.x = RoundFloatToInt(gaze_rect.x * w_ratio);
    gaze_rect.y = RoundFloatToInt(gaze_rect.y * h_ratio);
    gaze_rect.width = RoundFloatToInt(gaze_rect.width * w_ratio);
    gaze_rect.height = RoundFloatToInt(gaze_rect.height * h_ratio);

    for (uint y = 0; y < upscaled_frame_h; y += header_.transform_block_h) {
      for (uint x = 0; x < upscaled_frame_w; x += header_.transform_block_w) {
        std::vector<std::byte> raw_block;
        bool empty_and_reader_done = !in_blocks_.Pop(raw_block);
        if (empty_and_reader_done) {
          throw std::runtime_error{"failed to read all expected blocks"};
        }

        Block block;
        ParseBlock(raw_block, header_.channel_count, header_.transform_block_w,
                   header_.transform_block_h, block);

        cv::Rect roi(x, y, header_.transform_block_w,
                     header_.transform_block_h);
        cv::Mat3f decoded_block = upscaled_frame(roi);

        bool gazed = gaze_rect.contains(cv::Point2i(x, y));

        DecodeBlock(block, gazed, cfg_.foreground_quant_step,
                    cfg_.background_quant_step, decoded_block);
      }
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