#include "decoder.hpp"

#include <cassert>
#include <cstdio>

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

void DefaultInit(DecoderConfig* cfg) {
  assert(cfg);

  cfg->foreground_quant_step = 1;
  cfg->background_quant_step = 640;
  cfg->max_gaze_rect_w = 64;
  cfg->max_gaze_rect_h = 64;
}

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

Status Validate(DecoderConfig* cfg) {
  assert(cfg);

  if (cfg->foreground_quant_step == 0) {
    std::fprintf(stderr, "Foreground quantization step must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  if (cfg->background_quant_step == 0) {
    std::fprintf(stderr, "Background quantization step must be > 0.\n");
    return kStatus_InvalidParamError;
  }

  return kStatus_Ok;
}