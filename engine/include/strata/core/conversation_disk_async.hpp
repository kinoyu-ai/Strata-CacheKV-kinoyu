#pragma once
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_file.hpp"
#include <future>

namespace strata::core {
// One bounded, owning transfer buffer, never an additional conversation cache.
// Only the engine thread touches the cache/index. The worker sees host bytes only.
class ConversationDiskAsync {
    struct Result {
        SavedConversation key;
        std::filesystem::path dir;
        std::vector<size_t> sizes;
        uint64_t estimate = 0;
        bool mirror = false;
        std::string error;
    };
    ConversationDiskCache& cache_;
    std::future<Result> job_;
    std::function<void()> worker_start_;
public:
    explicit ConversationDiskAsync(ConversationDiskCache& cache, std::function<void()> worker_start = {})
        : cache_(cache), worker_start_(std::move(worker_start)) {}
    ~ConversationDiskAsync() { try { poll(true); } catch (...) {} }
    bool busy() const { return job_.valid(); }
    // Call from the engine thread, including while idle. Publication renames files,
    // then commits the existing durable group index; partial groups stay invisible.
    void poll(bool wait = false) {
        if (!job_.valid() || (!wait && job_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)) return;
        Result r;
        try { r = job_.get(); }
        catch (const std::exception& e) {
            std::error_code ec; std::filesystem::remove_all(cache_.staging_path(), ec);
            std::fprintf(stderr, "strata serve: async disk cache: worker failed: %s\n", e.what());
            return;
        }
        bool ok = false;
        if (r.error.empty()) {
            ok = cache_.store(r.key.live.ids, r.key.live.imgs, r.key.checkpoints, r.key.cvec,
                r.sizes.size(), r.estimate,
                [&](const std::string& dest, size_t k, size_t& bytes, std::string& error) {
                    std::error_code ec;
                    std::filesystem::rename(r.dir / (std::to_string(k) + ".bin"), dest, ec);
                    if (ec) { error = ec.message(); return false; }
                    bytes = r.sizes[k]; return true;
                }, r.error, r.mirror);
        }
        std::error_code ec;
        std::filesystem::remove_all(r.dir, ec);
        std::fprintf(stderr, "strata serve: async disk cache: %s %zu tokens%s%s\n",
            ok ? "committed" : "failed", r.key.live.ids.size(), r.error.empty() ? "" : "; ", r.error.c_str());
    }
    bool submit(SavedConversation image, SessionFileIdentity identity, uint64_t estimate,
                SessionWriteOptions options, bool mirror, std::string& error) {
        poll();
        if (busy()) { error = "previous write still in flight; skip snapshot"; return false; }
        if (!cache_.reserve(estimate, error)) return false;
        const auto dir = cache_.staging_path();
        try {
            std::filesystem::create_directory(dir);
            std::filesystem::permissions(dir, std::filesystem::perms::owner_all);
            job_ = std::async(std::launch::async,
                [image = std::move(image), identity, estimate, options = std::move(options), mirror, dir, start = worker_start_]() mutable {
                    if (start) start();
                    Result r; r.dir = dir; r.estimate = estimate; r.mirror = mirror;
                    r.key.live.ids = image.live.ids; r.key.live.imgs = image.live.imgs; r.key.cvec = image.cvec;
                    for (const auto& cp : image.checkpoints) {
                        r.key.checkpoints.emplace_back();
                        r.key.checkpoints.back().ids = cp.ids; r.key.checkpoints.back().imgs = cp.imgs;
                    }
                    try {
                        auto parts = std::move(image.stage_images);
                        const size_t count = parts.size() + 1;
                        for (size_t k = 0; k < count; ++k) {
                            auto& part = k ? parts[k-1] : image;
                            size_t bytes = 0;
                            if (!session_file_write((dir / (std::to_string(k) + ".bin")).string(),
                                    part, identity, bytes, r.error, options)) break;
                            r.sizes.push_back(bytes);
                            part = {}; // Release each stage's transfer buffer as soon as its file is written.
                        }
                        if (r.sizes.size() != count && r.error.empty()) r.error = "incomplete stage group";
                    } catch (const std::exception& e) { r.error = e.what(); }
                    std::fprintf(stderr, "strata serve: async disk cache: files ready %zu tokens%s%s\n",
                        r.key.live.ids.size(), r.error.empty() ? "" : "; ", r.error.c_str());
                    return r;
                });
            return true;
        } catch (const std::exception& e) {
            error = e.what(); std::error_code ec; std::filesystem::remove_all(dir, ec); return false;
        }
    }
};
}
