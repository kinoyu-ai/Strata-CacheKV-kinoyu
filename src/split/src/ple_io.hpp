// Optional bounded PLE reads for the split engine. Default keeps its locked RAM table.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace fnx {
void ple_io_configure(bool direct, uint64_t cache_rows,
                      const std::vector<std::pair<std::string,uint64_t>>& disks);
bool ple_io_direct();
void ple_io_open(const std::string& path, uint64_t offset, uint64_t rows, uint32_t row_bytes);
void ple_io_read(const uint32_t* rows, size_t count, uint8_t* out);
void ple_io_close();
}
