// Optional process-local second-level cache for immutable PLE source pages.
// Data lives on disk; only tags/checksums and one page of scratch live in RAM.
#pragma once
#include "strata/ngram/ple_disk_options.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::ngram {

class PleDiskCache {
public:
    static constexpr uint64_t page_bytes = 4096;
    static constexpr uint64_t chunk_bytes = 256ull << 20; // below FAT32's 4 GiB limit
    struct Stats { uint64_t hits = 0, misses = 0, writes = 0, corrupt = 0, failures = 0; };
    PleDiskCache() = default;
    PleDiskCache(const PleDiskCache&) = delete;
    PleDiskCache& operator=(const PleDiskCache&) = delete;
    ~PleDiskCache() { close(); }

    bool open(const std::vector<PleDiskLocation>& locations, std::string& error) {
        close(); stats_.reset();
#if defined(__linux__)
        try {
            for (const auto& loc : locations) {
                if (!loc.mib) continue;
                if (loc.mib > 1048576) return fail_open("capacity exceeds 1 TiB per directory", error);
                std::error_code ec;
                if (!std::filesystem::is_directory(loc.path, ec) || ec)
                    return fail_open("cache directory must already exist on the mounted drive", error);
                Root root;
                root.path = std::filesystem::canonical(loc.path, ec).string();
                if (ec) return fail_open("cannot resolve cache directory", error);
                struct stat st{};
                if (::lstat(root.path.c_str(), &st) || !S_ISDIR(st.st_mode) ||
                    st.st_uid != ::geteuid() || (st.st_mode & 0022))
                    return fail_open("cache directory must be owned by this user and not writable by others", error);
                roots_.push_back(std::move(root));
                Root& r = roots_.back();
                r.lock = ::open((r.path + "/.ple.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
                if (r.lock < 0 || ::flock(r.lock, LOCK_EX | LOCK_NB))
                    return fail_open("cache directory is locked or inaccessible", error);
                for (const auto& entry : std::filesystem::directory_iterator(r.path)) {
                    const std::string name = entry.path().filename().string();
                    if (name.rfind("ple-run-", 0) == 0 && !entry.is_symlink()) {
                        std::filesystem::remove_all(entry.path(), ec);
                        if (ec) return fail_open("cannot remove stale process cache", error);
                    }
                }
                std::string pattern = r.path + "/ple-run-XXXXXX";
                std::vector<char> temp(pattern.begin(), pattern.end()); temp.push_back(0);
                if (!::mkdtemp(temp.data())) return fail_open("cannot create process cache", error);
                r.run = temp.data();
                r.slots = loc.mib * (1ull << 20) / page_bytes;
                if (r.slots > UINT32_MAX) return fail_open("too many cache pages", error);
                r.entries.resize((size_t) r.slots);
                // Equal-budget roots are striped. Each root gets an independent hash table and byte cap.
                std::fprintf(stderr, "strata PLE disk cache: %s budget=%llu MiB, process-local, 256 MiB chunks\n",
                             r.path.c_str(), (unsigned long long) loc.mib);
            }
            enabled_ = !roots_.empty();
            return enabled_;
        } catch (const std::exception& e) { return fail_open(e.what(), error); }
#else
        (void) locations; error = "PLE disk cache requires Linux"; return false;
#endif
    }
    void close() noexcept {
#if defined(__linux__)
        if (!roots_.empty()) log();
        enabled_ = false;
        for (auto& r : roots_) {
            for (int fd : r.files) if (fd >= 0) ::close(fd);
            std::error_code ec;
            if (!r.run.empty()) std::filesystem::remove_all(r.run, ec);
            if (r.lock >= 0) { ::flock(r.lock, LOCK_UN); ::close(r.lock); }
        }
        roots_.clear();
#endif
    }
    void abandon(const char* reason) {
#if defined(__linux__)
        disable(reason);
#else
        (void) reason;
#endif
    }
    bool enabled() const { return enabled_; }
    uint64_t metadata_bytes() const {
        uint64_t n = 0;
#if defined(__linux__)
        for (const auto& r : roots_) n += r.entries.size() * sizeof(Entry);
#endif
        return n;
    }
    Stats stats() const { return {stats_.hits.load(), stats_.misses.load(), stats_.writes.load(), stats_.corrupt.load(), stats_.failures.load()}; }
    void log() const {
        const auto counters = stats();
        std::fprintf(stderr, "strata PLE disk cache: hits=%llu misses=%llu writes=%llu corrupt=%llu failures=%llu metadata=%llu KiB\n",
                     (unsigned long long) counters.hits, (unsigned long long) counters.misses,
                     (unsigned long long) counters.writes, (unsigned long long) counters.corrupt,
                     (unsigned long long) counters.failures, (unsigned long long) (metadata_bytes() >> 10));
    }
    // Cheap lookup before queueing I/O. A fill holding the stripe is a miss, never a stall for source reads.
    bool may_contain(uint64_t offset, uint32_t length) {
#if defined(__linux__)
        if (!enabled_ || offset % page_bytes || !length || length % page_bytes) return false;
        for (uint32_t at = 0; at < length; at += page_bytes) {
            const uint64_t page = (offset + at) / page_bytes;
            Root& r = roots_[page % roots_.size()];
            const size_t index = entry_index(r, page);
            std::unique_lock<std::mutex> guard(r.stripes[index % 1024], std::try_to_lock);
            if (!guard.owns_lock() || r.entries[index].page != page) { ++stats_.misses; return false; }
        }
        return true;
#else
        (void) offset; (void) length; return false;
#endif
    }
    // The caller owns an aligned buffer. `bytes` preserves EOF short reads from the immutable source.
    bool read(uint64_t offset, uint32_t length, uint8_t* dst, uint32_t& bytes) {
        bytes = 0;
#if defined(__linux__)
        if (!enabled_ || offset % page_bytes || !length || length % page_bytes) return false;
        for (uint32_t at = 0; at < length; at += page_bytes) {
            const uint64_t page = (offset + at) / page_bytes;
            Root& r = roots_[page % roots_.size()];
            const size_t index = entry_index(r, page);
            std::lock_guard<std::mutex> guard(r.stripes[index % 1024]);
            Entry& e = r.entries[index];
            if (e.page != page) { ++stats_.misses; return false; }
            const int fd = file(r, e.physical, false);
            if (fd < 0 || !transfer(fd, dst + at, page_bytes, slot_offset(e.physical), false)) {
                disable("cache read failed; using source"); return false;
            }
            if (checksum(dst + at) != e.checksum) {
                e.page = empty; ++stats_.corrupt; ++stats_.misses;
                std::fprintf(stderr, "strata PLE disk cache: checksum mismatch; using source\n");
                return false;
            }
            bytes += e.bytes;
            if (e.bytes < page_bytes) break;
        }
        ++stats_.hits;
        if ((stats_.hits.load() & 65535) == 0) log();
        return true;
#else
        (void) offset; (void) length; (void) dst; return false;
#endif
    }
    void write(uint64_t offset, uint32_t bytes, const uint8_t* src) {
#if defined(__linux__)
        if (!enabled_ || offset % page_bytes) return;
        for (uint32_t at = 0; at < bytes; at += page_bytes) {
            const uint64_t page = (offset + at) / page_bytes;
            Root& r = roots_[page % roots_.size()];
            const size_t index = entry_index(r, page);
            std::lock_guard<std::mutex> guard(r.stripes[index % 1024]);
            Entry& e = r.entries[index];
            if (e.page == page) continue;
            if (e.physical == UINT32_MAX) {
                std::lock_guard<std::mutex> files_guard(*r.file_mu);
                e.physical = (uint32_t) r.allocated++;
            }
            e.page = empty; // publish the tag only after the entire page is written
            const uint32_t n = std::min<uint32_t>(page_bytes, bytes - at);
            alignas(4096) uint8_t scratch_[4096];
            std::memcpy(scratch_, src + at, n);
            if (n < page_bytes) std::memset(scratch_ + n, 0, page_bytes - n);
            const int fd = file(r, e.physical, true);
            if (fd < 0 || !transfer(fd, scratch_, page_bytes, slot_offset(e.physical), true)) {
                disable("cache write failed or drive full; using source"); return;
            }
            e.checksum = checksum(scratch_); e.bytes = (uint16_t) n; e.page = page;
            ++stats_.writes;
        }
#else
        (void) offset; (void) bytes; (void) src;
#endif
    }
private:
    std::atomic<bool> enabled_{false};
    struct Counters {
        std::atomic<uint64_t> hits{0}, misses{0}, writes{0}, corrupt{0}, failures{0};
        void reset() { hits = 0; misses = 0; writes = 0; corrupt = 0; failures = 0; }
    } stats_;
#if defined(__linux__)
    static constexpr uint64_t empty = UINT64_MAX;
    struct Entry { uint64_t page = empty; uint32_t physical = UINT32_MAX, checksum = 0; uint16_t bytes = 0; };
    struct Root {
        std::string path, run;
        int lock = -1;
        uint64_t slots = 0, allocated = 0;
        std::vector<Entry> entries;
        std::vector<int> files;
        std::unique_ptr<std::mutex> file_mu = std::make_unique<std::mutex>();
        std::unique_ptr<std::mutex[]> stripes = std::make_unique<std::mutex[]>(1024);
    };
    std::vector<Root> roots_;
    size_t entry_index(const Root& r, uint64_t page) const {
        uint64_t x = page / roots_.size();
        x *= 0x9E3779B97F4A7C15ull; x ^= x >> 29;
        return (size_t) (x % r.slots);
    }
    static uint64_t slot_offset(uint32_t slot) { return (uint64_t) slot * page_bytes % chunk_bytes; }
    static uint32_t checksum(const uint8_t* p) {
        uint32_t h = 2166136261u;
        for (uint32_t i = 0; i < page_bytes; ++i) h = (h ^ p[i]) * 16777619u;
        return h;
    }
    int file(Root& r, uint32_t slot, bool create) {
        std::lock_guard<std::mutex> guard(*r.file_mu);
        const size_t chunk = (size_t) ((uint64_t) slot * page_bytes / chunk_bytes);
        if (r.files.size() <= chunk) r.files.resize(chunk + 1, -1);
        int& fd = r.files[chunk];
        if (fd >= 0 || !create) return fd;
        // Check space when opening each new 256 MiB chunk; never preallocate the whole budget.
        std::error_code ec;
        const auto space = std::filesystem::space(r.run, ec);
        if (ec || space.available < chunk_bytes + (128ull << 20)) return -1;
        const std::string path = r.run + "/pages-" + std::to_string(chunk) + ".bin";
        fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_DIRECT, 0600);
        // Some removable filesystems do not implement O_DIRECT. Refuse rather than filling the OS page cache.
        return fd;
    }
    static bool transfer(int fd, uint8_t* p, uint32_t n, uint64_t offset, bool writing) {
        // One aligned operation. A short transfer invalidates the entry; retrying a partial direct-I/O buffer
        // would break alignment. EINTR before a transfer is safe to retry.
        ssize_t done;
        do { done = writing ? ::pwrite(fd, p, n, (off_t) offset) : ::pread(fd, p, n, (off_t) offset); }
        while (done < 0 && errno == EINTR);
        return done == n;
    }
    void disable(const char* reason) {
        if (!enabled_.exchange(false)) return;
        ++stats_.failures;
        std::fprintf(stderr, "strata PLE disk cache: disabled: %s\n", reason);
    }
    bool fail_open(const std::string& why, std::string& out) {
        out = "PLE disk cache: " + why; close(); return false;
    }
#endif
};
} // namespace strata::ngram
