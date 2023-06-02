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
  cli::Opt opts[]{
      {"foreground-quant-step", cli::kOptArgType_Uint,
       &c->foreground_quant_step},
      {"background-quant-step", cli::kOptArgType_Uint,
       &c->background_quant_step},
      {"max-gaze-rect-w", cli::kOptArgType_Uint, &c->max_gaze_rect_w},
      {"max-gaze-rect-h", cli::kOptArgType_Uint, &c->max_gaze_rect_h}};

  uint argi;
  uint opts_size = sizeof(opts) / sizeof(opts[0]);

  cli::Status st = cli::ParseOpts(argc, argv, opts, opts_size, &argi);
  if (st != cli::kStatus_Ok) {
    std::fprintf(stderr, "parsing options: %s\n", cli::StatusMessage(st));
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