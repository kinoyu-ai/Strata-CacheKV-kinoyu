#pragma once
#include <cstdint>
#include <string>

namespace strata::ngram {
// One independently bounded disk-cache directory. No platform I/O is pulled into GPU headers.
struct PleDiskLocation { std::string path; uint64_t mib = 0; };
}
