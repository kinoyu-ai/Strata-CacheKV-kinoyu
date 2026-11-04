#include "strata/core/conversation_disk.hpp"
#include <fstream>
#include <sys/wait.h>
#include <cstdlib>
using namespace strata::core;
namespace fs = std::filesystem;
void check(bool ok, const char* msg) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); } }
int main() {
    const auto root = fs::temp_directory_path() / ("strata-disk-test-" + std::to_string(getpid()));
    fs::remove_all(root); fs::create_directory(root); fs::permissions(root, fs::perms::owner_all);
    fs::create_directory(root / "run-stale"); std::ofstream(root / "run-stale" / "partial") << "garbage";
    std::string error;
    auto writer = [](const std::string& file, size_t, size_t& bytes, std::string&) {
        std::ofstream f(file); f << std::string(16, 'x'); bytes = 16; return bool(f);
    };
    std::vector<int32_t> a(60001, 1), b(60001, 2), c(60001, 3);
    {
        ConversationDiskCache cache;
        check(cache.open(root.string(), 64, 60001, error), "open");
        check(!fs::exists(root / "run-stale"), "clean stale run");
        check(!cache.wants(60000) && cache.wants(60001), "60000/60001 boundary");
        check(!cache.store(std::vector<int32_t>(60000), {}, {}, true, 2, 32, writer, error), "short not stored");
        check(cache.store(a, {}, {}, true, 2, 32, writer, error), "store two stages");
        check(cache.size() == 1 && cache.bytes() == 32, "atomic group bytes");
        auto more_a = a; more_a.push_back(9);
        auto ma = cache.best(more_a, {}, true);
        check(ma.tokens == 60001 && ma.live, "resume live prefix");
        check(!cache.best(more_a, {}, false).tokens, "steering isolation");
        check(!cache.best(more_a, {{1,77}}, true).tokens, "image isolation");
        check(cache.store(b, {}, {}, true, 2, 32, writer, error), "store B");
        cache.touch(ma.serial);
        check(cache.store(c, {}, {}, true, 2, 32, writer, error), "store C evicts B");
        auto more_b = b; more_b.push_back(9);
        check(cache.best(more_a, {}, true).tokens && !cache.best(more_b, {}, true).tokens, "LRU uses last access");
        auto failure = [&](const std::string& f, size_t k, size_t& n, std::string& e) {
            if (k) { e = "injected second stage write failure"; return false; }
            return writer(f,k,n,e);
        };
        cache.clear();
        check(!cache.store(a, {}, {}, true, 2, 32, failure, error) && cache.size() == 0, "partial group not published");
        for (const auto& item : fs::directory_iterator(root / ("run-" + std::to_string(getpid()))))
            check(false, "partial files cleaned");
        check(cache.store(a, {}, {}, true, 2, 32, writer, error), "store before guard");
        check(cache.store(b, {}, {}, true, 2, 32, writer, error), "store B before guard");
        cache.guard(cache.best(more_a, {}, true).serial);
        check(!cache.store(c, {}, {}, true, 2, 64, writer, error), "guard keeps incoming files");
        cache.unguard();
        ConversationDiskCache other; std::string why;
        check(!other.open(root.string(),64,60001,why), "second engine cannot purge active cache");
        check(cache.best(more_a, {}, true).tokens, "locked cache remains usable");
    }
    check(!fs::exists(root / ("run-" + std::to_string(getpid()))), "normal shutdown cleans run");
    // A process killed by _exit skips destructors. The next lock owner discards its files.
    auto child = fork();
    if (!child) {
        ConversationDiskCache cache; std::string why;
        if (!cache.open(root.string(),64,60001,why) || !cache.store(a,{}, {},true,2,32,writer,why)) _exit(1);
        _exit(0);
    }
    int status = 0; waitpid(child,&status,0); check(status == 0, "crashed child fixture");
    {
        ConversationDiskCache cache; check(cache.open(root.string(),64,60001,error), "restart after crash");
        check(cache.size() == 0 && !fs::exists(root / ("run-" + std::to_string(child))), "crash cache discarded");
    }
    fs::remove_all(root);
    std::puts("conversation_disk_test passed");
}
