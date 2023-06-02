#include "decoder.hpp"

#include <cassert>
#include <cstdio>

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

void DefaultInit(DecoderConfig* c) {
  assert(c);

  c->foreground_quant_step = 1;
  c->background_quant_step = 640;
  c->max_gaze_rect_w = 64;
  c->max_gaze_rect_h = 64;
}

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

Status Validate(DecoderConfig* c) {
  assert(c);

  if (c->foreground_quant_step == 0) {
    std::fprintf(stderr, "Foreground quantization step must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (c->background_quant_step == 0) {
    std::fprintf(stderr, "Background quantization step must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  return kStatus_Ok;
}