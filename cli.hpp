#ifndef SCALABLE_VIDEO_CODEC_CLI_HPP
#define SCALABLE_VIDEO_CODEC_CLI_HPP

#include "types.hpp"

enum OptionType {
  kOptionTypeInt,
  kOptionTypeUInt,
  kOptionTypeBoolean,
  kOptionTypeFloat
};

#define OPTION_NAME_BUF_SZ 64

struct Option {
  char name[OPTION_NAME_BUF_SZ];
  OptionType type;
  void* val;
};

enum CliStatus {
  kCliStatusOk,
  kCliStatusInvalidOptionType,
  kCliStatusMissingOptionValue,
  kCliStatusInvalidOptionValue,
  kCliStatusUnexpectedOptionName,
};

const char* CliStatusMessage(CliStatus cs);

/*
Parses options from arguments passed to the program.

Each option provided at the command-line has its name prefixed with "--" and has
an associated argument following its name. `--kmeans-cluster-count 10` is an
example.

Options must be before positional parameters. Otherwise, the intended
options would be considered positional parameters. For instance, in `encode
--kmeans-cluster-count 10 foreman.mp4`, the option `--kmeans-cluster-count 10`
is before the positional parameter `foreman.mp4`.

Input Parameters:
- argc:  number of arguments passed to the program
- argv: all arguments passed to the program
- opts_size: number of options

Output Parameters:
- opts: options
- argi: argument index that is one past that of the last option successfully
parsed. cannot be null
*/
CliStatus ParseOptions(uint argc, char* argv[], uint opts_size, Option opts[],
                       uint* argi);

#endif  // SCALABLE_VIDEO_CODEC_CLI_HPP