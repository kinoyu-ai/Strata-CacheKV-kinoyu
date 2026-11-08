#include "ple_io.hpp"
#include "strata/ngram/ple_reader.hpp"
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
namespace fnx {
namespace {
struct State {
    std::mutex mutex;
    bool direct = false;
    uint64_t cache_rows = 65536;
    std::vector<strata::ngram::PleDiskLocation> disks;
    strata::ngram::PleReader reader;
};
State& state() { static State s; return s; }
}
void ple_io_configure(bool direct, uint64_t cache_rows,
                      const std::vector<std::pair<std::string,uint64_t>>& disks) {
    State& s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    s.reader.close(); s.direct=direct; s.cache_rows=cache_rows; s.disks.clear();
    if (!direct && !disks.empty()) throw std::runtime_error("--ple-disk-cache needs --ple-io direct");
    for (const auto& d:disks) s.disks.push_back({d.first,d.second});
}
bool ple_io_direct() { return state().direct; }
void ple_io_open(const std::string& path,uint64_t offset,uint64_t rows,uint32_t row_bytes) {
    State& s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.direct) throw std::runtime_error("PLE direct reader is not configured");
    if (!rows || rows > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("PLE direct row count is out of range");
    std::string error;
    if (!s.reader.open(path,offset,rows,128,s.cache_rows,error,true,row_bytes,s.disks))
        throw std::runtime_error("PLE direct open: "+error);
    std::fprintf(stderr,"fnx: PLE direct reader: %llu rows, %u bytes/row, RAM row cache %llu; table is not locked in RAM\n",
                 (unsigned long long)rows,row_bytes,(unsigned long long)s.cache_rows);
}
void ple_io_read(const uint32_t* rows,size_t count,uint8_t* out) {
    State& s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.reader.is_open()) throw std::runtime_error("PLE direct reader is not open");
    std::string error;auto ticket=s.reader.issue(rows,count,out);
    if (!s.reader.collect(ticket,error)) throw std::runtime_error("PLE direct read: "+error);
}
void ple_io_close() {
    State& s=state(); std::lock_guard<std::mutex> lock(s.mutex);s.reader.close();
}
}
