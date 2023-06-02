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

  Status status = kStatus_InvalidParamError;

  if (c->foreground_quant_step == 0) {
    std::fprintf(stderr, "foreground quantization step must be > 0\n");
    return status;
  }

  if (c->background_quant_step == 0) {
    std::fprintf(stderr, "background quantization step must be > 0\n");
    return status;
  }

  status = kStatus_Ok;

  return status;
}