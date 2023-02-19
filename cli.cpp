#include "cli.hpp"

#include <cstdio>
#include <cstring>

// Returns 1 if `str` is prefixed by `prefix`. Otherwise, returns 0.
static boolean HasPrefix(const char* str, const char* prefix) {
  return std::strncmp(str, prefix, std::strlen(prefix)) == 0;
}

const char* CliStatusMessages[] = {
    "success", "invalid option type", "missing option value",
    "invalid option value", "unexpected option name"};

const char* CliStatusMessage(CliStatus cs) { return CliStatusMessages[cs]; }

CliStatus ParseOptions(uint argc, char* argv[], uint opts_size, Option opts[],
                       uint* argi) {
  uint i = 1;

  while (i < argc && HasPrefix(argv[i], "--")) {
    uint j = 0;

    while (j < opts_size && std::strcmp(&argv[i][2], opts[j].name) != 0) {
      ++j;
    }

    if (j == opts_size) {
      *argi = i;
      return kCliStatusUnexpectedOptionName;
    }

    if (i + 1 >= argc) {
      *argi = i;
      return kCliStatusMissingOptionValue;
    }

    const char* format;
    switch (opts[j].type) {
      case kOptionTypeBoolean:
      case kOptionTypeInt: {
        format = "%d";
        break;
      }
      case kOptionTypeUInt: {
        format = "%u";
        break;
      }
      case kOptionTypeFloat: {
        format = "%f";
        break;
      }
      default: {
        *argi = i;
        return kCliStatusInvalidOptionType;
      }
    }

    int ret = std::sscanf(argv[i + 1], format, opts[j].val);
    if (ret != 1) {
      *argi = i;
      return kCliStatusInvalidOptionValue;
    }
    if (opts[j].type == kOptionTypeBoolean) {
      if (*(boolean*)opts[j].val > 1) {
        *argi = i;
        return kCliStatusInvalidOptionValue;
      }
    }

    i += 2;
  }

  *argi = i;

  return kCliStatusOk;
}