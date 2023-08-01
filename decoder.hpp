#ifndef SCALABLE_VIDEO_CODEC_DECODER_HPP
#define SCALABLE_VIDEO_CODEC_DECODER_HPP

#include <cstddef>
#include <vector>

#include "codec.hpp"
#include "error.hpp"
#include "queue.hpp"
#include "types.hpp"

struct DecoderConfig {
  uint foreground_quant_step;
  uint background_quant_step;
  uint max_gaze_rect_w;
  uint max_gaze_rect_h;
};

Error Validate(DecoderConfig&);

class Decoder {
 public:
  Decoder(const DecoderConfig& cfg, const Header& header,
          CircularQueue<std::vector<std::byte>>& in_blocks);
  void operator()();

 private:
  DecoderConfig cfg_;
  Header header_;
  CircularQueue<std::vector<std::byte>>& in_blocks_;
};

#endif  // SCALABLE_VIDEO_CODEC_DECODER_HPP