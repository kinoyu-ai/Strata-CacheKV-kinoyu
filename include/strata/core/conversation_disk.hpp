#pragma once
// Process-local, disposable disk tier. Only token/image matching metadata stays in RAM.
#include "strata/core/conversation_cache.hpp"
#include <filesystem>
#include <functional>
#include <cstdio>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {
class ConversationDiskCache {
public:
    struct Entry {
        uint64_t serial = 0, used = 0, bytes = 0;
        bool cvec = true;
        ConversationCheckpoint live;
        std::vector<ConversationCheckpoint> checkpoints;
        std::vector<std::string> files;
        std::filesystem::path dir;
    };
    struct Match { uint64_t serial = 0; int64_t tokens = 0; bool live = false; };
    ConversationDiskCache() = default;
    ConversationDiskCache(const ConversationDiskCache&) = delete;
    ConversationDiskCache& operator=(const ConversationDiskCache&) = delete;
    ~ConversationDiskCache() { close(); }
    bool open(const std::string& root, uint64_t budget, int64_t min_tokens, std::string& error) {
        if (root.empty() || !budget) return true;
#if defined(__linux__)
        try {
            namespace fs = std::filesystem;
            fs::path base(root);
            if (fs::create_directories(base)) fs::permissions(base, fs::perms::owner_all);
            struct stat st{};
            if (::lstat(root.c_str(), &st) || !S_ISDIR(st.st_mode) || st.st_uid != ::getuid() || (st.st_mode & 0022))
                throw std::runtime_error("cache root must be an owned directory without group/other write access");
            lock_ = ::open((base / ".lock").c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
            if (lock_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB))
                throw std::runtime_error("cache root is unavailable or already used by another engine");
            for (const auto& item : fs::directory_iterator(base))
                if (item.path().filename().string().rfind("run-", 0) == 0) {
                    fs::remove_all(item.path());
                    std::fprintf(stderr, "strata serve: disk cache: removed stale %s\n", item.path().filename().c_str());
                }
            run_ = base / ("run-" + std::to_string(::getpid()));
            fs::create_directory(run_);
            fs::permissions(run_, fs::perms::owner_all);
            budget_ = budget; min_tokens_ = min_tokens;
            return true;
        } catch (const std::exception& e) { error = e.what(); close(); return false; }
#else
        error = "the disposable disk cache currently requires Linux";
        return false;
#endif
    }
    bool enabled() const { return budget_ != 0; }
    bool wants(size_t n) const { return enabled() && n >= (uint64_t) min_tokens_; }
    uint64_t bytes() const { return bytes_; }
    size_t size() const { return entries_.size(); }
    const Entry* find(uint64_t serial) const {
        for (const auto& e : entries_) if (e.serial == serial) return &e;
        return nullptr;
    }
    template<class Token> Match best(const std::vector<Token>& ids, const std::vector<ConversationImageKey>& imgs,
                                     bool cvec) const {
        Match m; uint64_t last = 0;
        if (!enabled()) return m;
        for (const auto& e : entries_) {
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& cp, bool live) {
                const auto n = conversation_prefix(cp, ids, imgs);
                if (n > m.tokens || (n > 0 && n == m.tokens && e.used > last)) {
                    m = {e.serial, n, live}; last = e.used;
                }
            };
            consider(e.live, true);
            for (const auto& cp : e.checkpoints) consider(cp, false);
        }
        return m;
    }
    void guard(uint64_t serial) { guarded_ = serial; }
    void unguard() { guarded_ = 0; }
    void touch(uint64_t serial) { for (auto& e : entries_) if (e.serial == serial) e.used = ++clock_; }
    void erase(uint64_t serial, const char* reason) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) if (it->serial == serial) {
            std::error_code ec;
            std::filesystem::remove_all(it->dir, ec);
            if (ec) {
                budget_ = 0;
                std::fprintf(stderr, "strata serve: disk cache disabled: cannot remove entry (%s)\n", ec.message().c_str());
            }
            std::fprintf(stderr, "strata serve: disk cache: removed %zu tokens, %llu bytes (%s)\n",
                         it->live.ids.size(), (unsigned long long) it->bytes, reason);
            bytes_ -= it->bytes; entries_.erase(it); return;
        }
    }
    void clear() { while (!entries_.empty()) erase(entries_.back().serial, "flush"); }
    // writer(file, stage, bytes, error). No entry is visible until every stage succeeds.
    bool store(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
               const std::vector<ConversationCheckpoint>& checks, bool cvec, size_t stages, uint64_t estimate,
               const std::function<bool(const std::string&, size_t, size_t&, std::string&)>& writer,
               std::string& error) {
        if (!wants(ids.size())) return false;
        std::filesystem::path pending;
        try {
            if (estimate > budget_) { error = "snapshot exceeds disk budget"; return false; }
            while (bytes_ > budget_ - estimate && !entries_.empty()) {
                auto oldest = entries_.end();
                for (auto it = entries_.begin(); it != entries_.end(); ++it)
                    if (it->serial != guarded_ && (oldest == entries_.end() || it->used < oldest->used)) oldest = it;
                if (oldest == entries_.end()) { error = "no room beside incoming conversation"; return false; }
                erase(oldest->serial, "LRU capacity");
                if (!enabled()) { error = "disk eviction failed"; return false; }
            }
            Entry e; e.serial = ++serial_; e.used = ++clock_; e.cvec = cvec;
            e.live.ids = ids; e.live.imgs = imgs;
            if (const auto* cp = session_checkpoint(checks)) {
                e.checkpoints.emplace_back(); e.checkpoints.back().ids = cp->ids; e.checkpoints.back().imgs = cp->imgs;
            }
            pending = run_ / std::to_string(e.serial);
            std::filesystem::create_directory(pending); e.dir = pending;
            for (size_t k = 0; k < stages; ++k) {
                auto path = (pending / ("stage-" + std::to_string(k) + ".bin")).string();
                size_t bytes = 0;
                if (!writer(path, k, bytes, error)) throw std::runtime_error(error);
                if (bytes > estimate - e.bytes) throw std::runtime_error("snapshot exceeded reservation");
                e.bytes += bytes; e.files.push_back(std::move(path));
            }
            // Replace an identical or strictly extended saved branch only after the new one is complete.
            std::vector<uint64_t> replaced;
            for (const auto& old : entries_) if (old.cvec == cvec && old.serial != guarded_) {
                auto covered = [&](const ConversationCheckpoint& cp) {
                    return conversation_covers(e.live, cp.ids, cp.imgs) ||
                        std::any_of(e.checkpoints.begin(), e.checkpoints.end(), [&](const auto& c) {
                            return conversation_covers(c, cp.ids, cp.imgs);
                        });
                };
                if (covered(old.live) && std::all_of(old.checkpoints.begin(), old.checkpoints.end(), covered))
                    replaced.push_back(old.serial);
            }
            for (auto serial : replaced) erase(serial, "superseded");
            if (!enabled()) throw std::runtime_error("disk eviction failed");
            const auto added = e.bytes;
            entries_.push_back(std::move(e)); bytes_ += added;
            return true;
        } catch (const std::exception& e) {
            error = e.what();
            if (!pending.empty()) { std::error_code ec; std::filesystem::remove_all(pending, ec); }
            return false;
        }
    }
private:
    static const ConversationCheckpoint* session_checkpoint(const std::vector<ConversationCheckpoint>& checks) {
        const ConversationCheckpoint* cp = nullptr;
        for (const auto& c : checks) if (!cp || c.ids.size() > cp->ids.size()) cp = &c;
        return cp;
    }
    void close() noexcept {
        entries_.clear();
        if (!run_.empty()) { std::error_code ec; std::filesystem::remove_all(run_, ec); run_.clear(); }
#if defined(__linux__)
        if (lock_ >= 0) { ::close(lock_); lock_ = -1; }
#endif
        budget_ = bytes_ = 0;
    }
    int lock_ = -1;
    std::filesystem::path run_;
    std::vector<Entry> entries_;
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0, clock_ = 0, guarded_ = 0;
    int64_t min_tokens_ = 60001;
};
} // namespace strata::core
