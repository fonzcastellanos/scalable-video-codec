#include "decoder_config.hpp"

#include <cassert>
#include <cstdio>

#include "cli.hpp"

Status ParseConfig(int argc, char* argv[], Config* c) {
  assert(argv);
  assert(c);

  Status status = kStatus_InvalidParamError;

  /*******************************************************************************
   * Command-line Options    #options
   *******************************************************************************/
  Option opts[]{
      {"foreground-quant-step", kOptionTypeUInt, &c->foreground_quant_step},
      {"background-quant-step", kOptionTypeUInt, &c->background_quant_step},
      {"max-gaze-rect-w", kOptionTypeUInt, &c->max_gaze_rect_w},
      {"max-gaze-rect-h", kOptionTypeUInt, &c->max_gaze_rect_h}};

  uint argi;
  uint opts_size = sizeof(opts) / sizeof(opts[0]);

  CliStatus st = ParseOptions(argc, argv, opts_size, opts, &argi);
  if (st != kCliStatusOk) {
    std::fprintf(stderr, "parsing options: %s\n", CliStatusMessage(st));
    return status;
  }

  status = kStatus_Ok;

  return status;
}

/*******************************************************************************
 * Default Config Values    #default-cfg
 *******************************************************************************/

void DefaultInit(Config* c) {
  assert(c);

  c->foreground_quant_step = 1;
  c->background_quant_step = 640;
  c->max_gaze_rect_w = 64;
  c->max_gaze_rect_h = 64;
}

/*******************************************************************************
 * Config Validation Functions    #cfg-validation
 *******************************************************************************/

Status Validate(Config* c) {
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