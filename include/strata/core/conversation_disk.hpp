#pragma once
// Disposable by default; optional restart-persistent tier. Only matching metadata stays in RAM.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_disk_index.hpp"
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
    bool open(const std::string& root, uint64_t budget, int64_t min_tokens, std::string& error,
              bool persistent = false, uint64_t compatibility = 0, uint64_t max_tokens = 1048576) {
        close();
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
            persistent_ = persistent; compatibility_ = compatibility; max_tokens_ = max_tokens;
            run_ = base / (persistent ? "persistent" : "run-" + std::to_string(::getpid()));
            if (fs::exists(fs::symlink_status(run_))) {
                struct stat rs{};
                if (::lstat(run_.c_str(), &rs) || !S_ISDIR(rs.st_mode) || rs.st_uid != ::getuid() || (rs.st_mode & 0022))
                    throw std::runtime_error("persistent cache directory ownership/type");
            } else if (!fs::create_directory(run_)) throw std::runtime_error("create cache directory");
            fs::permissions(run_, fs::perms::owner_all);
            budget_ = budget; min_tokens_ = min_tokens;
            if (persistent_) { disk_index::sync_dir(base); recover(); }
            return true;
        } catch (const std::exception& e) { error = e.what(); close(); return false; }
#else
        error = "the disposable disk cache currently requires Linux";
        return false;
#endif
    }
    bool persistent() const { return persistent_; }
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
    void touch(uint64_t serial) {
        for (auto& e : entries_) if (e.serial == serial) {
            e.used = ++clock_;
            if (persistent_) try { write_index(e); } catch(const std::exception& x) {
                std::fprintf(stderr,"strata serve: disk cache: LRU index update failed: %s\n",x.what());
            }
        }
    }
    void set_prefer_keep(std::function<bool(const Entry&)> keep) { prefer_keep_ = std::move(keep); }
    bool contains(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                  const std::vector<ConversationCheckpoint>& checks, bool cvec) {
        const auto* cp = session_checkpoint(checks);
        for (auto& e : entries_) if(e.cvec==cvec && e.live.ids==ids && e.live.imgs==imgs &&
            ((!cp && e.checkpoints.empty()) || (cp && e.checkpoints.size()==1 &&
             e.checkpoints[0].ids==cp->ids && e.checkpoints[0].imgs==cp->imgs))) {
            touch(e.serial); return true;
        }
        return false;
    }
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
            bytes_ -= it->bytes; entries_.erase(it);
            if (persistent_ && !ec) try { disk_index::sync_dir(run_); } catch (const std::exception& x) {
                std::fprintf(stderr,"strata serve: disk cache: deletion flush failed: %s\n",x.what());
            }
            return;
        }
    }
    void clear() { while (!entries_.empty()) erase(entries_.back().serial, "flush"); }
    // writer(file, stage, bytes, error). No entry is visible until every stage succeeds.
    bool store(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
               const std::vector<ConversationCheckpoint>& checks, bool cvec, size_t stages, uint64_t estimate,
               const std::function<bool(const std::string&, size_t, size_t&, std::string&)>& writer,
               std::string& error, bool mirror = false) {
        if (!enabled() || (!mirror && !wants(ids.size()))) return false;
        if (persistent_ && contains(ids, imgs, checks, cvec)) return true;
        std::filesystem::path pending;
        try {
            if (estimate > budget_) { error = "snapshot exceeds disk budget"; return false; }
            while (bytes_ > budget_ - estimate && !entries_.empty()) {
                auto oldest = entries_.end();
                for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                    if (it->serial == guarded_) continue;
                    const bool keep = prefer_keep_ && prefer_keep_(*it);
                    const bool old_keep = oldest != entries_.end() && prefer_keep_ && prefer_keep_(*oldest);
                    if (oldest == entries_.end() || (old_keep && !keep) || (keep == old_keep && it->used < oldest->used)) oldest = it;
                }
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
            if (!std::filesystem::create_directory(pending)) throw std::runtime_error("cache entry directory exists");
            std::filesystem::permissions(pending,std::filesystem::perms::owner_all); e.dir = pending;
            for (size_t k = 0; k < stages; ++k) {
                auto path = (pending / ("stage-" + std::to_string(k) + ".bin")).string();
                size_t bytes = 0;
                if (!writer(path, k, bytes, error)) throw std::runtime_error(error);
                if (bytes > estimate - e.bytes) throw std::runtime_error("snapshot exceeded reservation");
                e.bytes += bytes; e.files.push_back(std::move(path));
            }
            if (persistent_) {
                const auto index_size = index_bytes(e).size();
                if (index_size > estimate - e.bytes) throw std::runtime_error("disk index exceeded reservation");
                e.bytes += index_size;
                write_index(e);
                disk_index::sync_dir(run_);
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
        if (!run_.empty()) {
            if (!persistent_) { std::error_code ec; std::filesystem::remove_all(run_, ec); }
            run_.clear();
        }
#if defined(__linux__)
        if (lock_ >= 0) { ::close(lock_); lock_ = -1; }
#endif
        budget_ = bytes_ = serial_ = clock_ = guarded_ = 0;
        persistent_ = false; prefer_keep_ = {};
    }
    std::vector<uint8_t> index_bytes(const Entry& e) const {
        disk_index::Codec c;
        c.put(0x31584944564b5453ull); c.put(compatibility_); c.put(e.serial); c.put(e.used);
        c.put(e.cvec ? 1 : 0); c.put(e.files.size()); c.put(e.bytes);
        c.checkpoint(e.live); c.put(e.checkpoints.size());
        for(const auto& cp:e.checkpoints)c.checkpoint(cp);
        c.put(disk_index::hash(c.b,c.b.size()));return std::move(c.b);
    }
    void write_index(const Entry& e) const {
#if defined(__linux__)
        disk_index::write(e.dir / "index.bin",index_bytes(e));
#else
        (void)e;
#endif
    }
    void recover() {
#if defined(__linux__)
        namespace fs=std::filesystem;
        uint64_t metadata=0;
        for(const auto& item:fs::directory_iterator(run_)) {
            try {
                struct stat st{};
                if(::lstat(item.path().c_str(),&st)||!S_ISDIR(st.st_mode)||st.st_uid!=::getuid()||(st.st_mode&0022))
                    throw std::runtime_error("entry type/owner");
                auto c=disk_index::read(item.path()/"index.bin");
                const auto index_size=c.b.size();
                if(metadata+index_size>disk_index::max_bytes || entries_.size()>=4096)
                    throw std::runtime_error("index memory limit");
                c.at=c.b.size()-8;const auto checksum=c.get();c.b.resize(c.b.size()-8);c.at=0;
                if(disk_index::hash(c.b,c.b.size())!=checksum)throw std::runtime_error("index checksum");
                if(c.get()!=0x31584944564b5453ull || c.get()!=compatibility_)throw std::runtime_error("different model/build/configuration");
                Entry e;e.serial=c.get();e.used=c.get();const auto mode=c.get();const auto stages=c.get();e.bytes=c.get();
                if(!e.serial || e.serial==UINT64_MAX || e.used==UINT64_MAX || mode>1 || stages<1 || stages>8 ||
                   item.path().filename()!=std::to_string(e.serial))throw std::runtime_error("index geometry");
                e.cvec=mode!=0;e.dir=item.path();c.checkpoint(e.live,max_tokens_);
                if(e.live.ids.empty())throw std::runtime_error("empty entry");
                auto n=c.get();if(n>1)throw std::runtime_error("index checkpoint count");e.checkpoints.resize((size_t)n);
                for(auto& cp:e.checkpoints)c.checkpoint(cp,max_tokens_);
                if(c.at!=c.b.size())throw std::runtime_error("trailing disk index bytes");
                uint64_t total=index_size;
                for(size_t k=0;k<stages;++k) {
                    auto path=e.dir/("stage-"+std::to_string(k)+".bin");struct stat part{};
                    if(::lstat(path.c_str(),&part)||!S_ISREG(part.st_mode)||part.st_nlink!=1||part.st_uid!=::getuid()||
                       (part.st_mode&0022)||part.st_size<=0 || (uint64_t)part.st_size>UINT64_MAX-total)
                        throw std::runtime_error("missing/invalid stage file");
                    total+=(uint64_t)part.st_size;e.files.push_back(path.string());
                }
                for(const auto& file:fs::directory_iterator(e.dir)) {
                    const auto name=file.path().filename().string();
                    if(name.rfind(".index-",0)==0) {fs::remove_all(file.path());continue;}
                    if(name!="index.bin" && std::find(e.files.begin(),e.files.end(),file.path().string())==e.files.end())
                        throw std::runtime_error("uncommitted file in entry");
                }
                if(total!=e.bytes || total>budget_)throw std::runtime_error("entry size/budget");
                if(total>UINT64_MAX-bytes_)throw std::runtime_error("cache size overflow");
                bytes_+=total;serial_=std::max(serial_,e.serial);clock_=std::max(clock_,e.used);
                metadata+=index_size;entries_.push_back(std::move(e));
            } catch(const std::exception& x) {
                std::fprintf(stderr,"strata serve: persistent disk cache: discard %s: %s\n",item.path().filename().c_str(),x.what());
                fs::remove_all(item.path());
            }
        }
        while(bytes_>budget_ && !entries_.empty()) {
            auto oldest=std::min_element(entries_.begin(),entries_.end(),[](const auto& a,const auto& b){return a.used<b.used;});
            erase(oldest->serial,"startup LRU capacity");
        }
        std::fprintf(stderr,"strata serve: persistent disk cache: indexed %zu entries, %llu bytes; payloads validated on reuse\n",
                     entries_.size(),(unsigned long long)bytes_);
#endif
    }
    bool persistent_ = false;
    uint64_t compatibility_ = 0, max_tokens_ = 1048576;
    std::function<bool(const Entry&)> prefer_keep_;
    int lock_ = -1;
    std::filesystem::path run_;
    std::vector<Entry> entries_;
    uint64_t budget_ = 0, bytes_ = 0, serial_ = 0, clock_ = 0, guarded_ = 0;
    int64_t min_tokens_ = 60001;
};
} // namespace strata::core
