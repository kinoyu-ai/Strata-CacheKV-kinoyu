#include "strata/ngram/ple_reader.hpp"
#include "strata/ngram/ple_disk_cache.hpp"
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; std::exit(1); } } while (0)

using namespace strata::ngram;
namespace fs = std::filesystem;
static fs::path temp(const fs::path& parent) {
    std::string s = (parent / "ple-test-XXXXXX").string();
    std::vector<char> p(s.begin(), s.end()); p.push_back(0);
    char* result = ::mkdtemp(p.data()); CHECK(result); return result;
}
static fs::path chunk(const fs::path& root) {
    for (const auto& r : fs::directory_iterator(root))
        if (r.is_directory()) for (const auto& f : fs::directory_iterator(r.path()))
            if (f.path().extension() == ".bin") return f.path();
    return {};
}
static void core(const fs::path& a, const fs::path& b) {
    alignas(4096) std::array<uint8_t, 8192> data{}, out{};
    for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) (i * 31 + i / 4096);
    std::string err; PleDiskCache c;
    CHECK(c.open({{a.string(), 1}, {b.string(), 1}}, err));
    uint32_t bytes;
    CHECK(!c.read(0, 8192, out.data(), bytes));
    c.write(0, 8192, data.data()); CHECK(c.enabled());
    CHECK(c.read(0, 8192, out.data(), bytes)); CHECK(bytes == 8192 && out == data);
    CHECK(!chunk(a).empty() && !chunk(b).empty());
    PleDiskCache second;
    CHECK(!second.open({{a.string(), 1}}, err)); // never clears another engine's pages
    CHECK(c.read(0, 8192, out.data(), bytes));
    // Corrupt a cached page. It must miss, then accept a freshly fetched source page.
    { std::fstream f(chunk(a), std::ios::binary | std::ios::in | std::ios::out); char x = 42; f.write(&x, 1); }
    CHECK(!c.read(0, 4096, out.data(), bytes)); CHECK(c.stats().corrupt == 1);
    c.write(0, 4096, data.data()); CHECK(c.read(0, 4096, out.data(), bytes));
    // EOF page: preserve valid length, zero padding, never invent source bytes.
    c.write(2 * 4096, 90, data.data()); CHECK(c.read(2 * 4096, 4096, out.data(), bytes));
    CHECK(bytes == 90 && !std::memcmp(out.data(), data.data(), 90));
    // Saturate both byte budgets, forcing hashed eviction. No hit may return another page's bytes.
    for (uint32_t p = 10; p < 1034; ++p) { std::memcpy(data.data(), &p, sizeof(p)); c.write((uint64_t) p * 4096, 4096, data.data()); }
    for (uint32_t p = 10; p < 1034; ++p)
        if (c.read((uint64_t) p * 4096, 4096, out.data(), bytes)) { uint32_t got; std::memcpy(&got, out.data(), 4); CHECK(got == p); }
    for (const auto& root : {a,b}) {
        uint64_t n = 0; for (auto const& x : fs::recursive_directory_iterator(root)) if (x.is_regular_file()) n += x.file_size();
        CHECK(n <= 1ull << 20);
    }
    // A broken cache read disables only the tier. The source reader is still usable.
    const auto file = chunk(a); fs::resize_file(file, 0);
    for (uint32_t p = 10; p < 1034 && c.enabled(); ++p) c.read((uint64_t) p * 4096, 4096, out.data(), bytes);
    CHECK(!c.enabled() && c.stats().failures == 1);
    c.close(); CHECK(chunk(a).empty() && chunk(b).empty());
    // Crash residue is discarded on the next exclusive open.
    pid_t pid = ::fork(); CHECK(pid >= 0);
    if (!pid) { PleDiskCache child; std::string why; if (!child.open({{a.string(), 1}}, why)) ::_exit(2); child.write(0, 4096, data.data()); ::_exit(0); }
    int status; CHECK(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(!chunk(a).empty()); CHECK(c.open({{a.string(), 1}}, err)); CHECK(chunk(a).empty()); c.close();
    CHECK(!c.open({{a.string(), 1}, {a.string(), 1}}, err)); // duplicate root rejected
    CHECK(c.open({{a.string(), 1}}, err)); c.close();
}
static void reader(const fs::path& base, const fs::path& a, const fs::path& b, uint32_t rb, bool threaded) {
    const auto source = base / "source.bin";
    const uint64_t offset = 192, rows = 2000;
    std::vector<uint8_t> contents(offset + rows * rb);
    std::mt19937 random(42); for (auto& x : contents) x = (uint8_t) random();
    { std::ofstream f(source, std::ios::binary); f.write((char*) contents.data(), contents.size()); }
    PleReader r; std::string err;
    CHECK(r.open(source.string(), offset, rows, 16, 0, err, threaded, rb, {{a.string(), 2}, {b.string(), 2}}));
    std::vector<uint32_t> ids{0, 1, 42, 43, 44, 45, 100, 1999, 2000, 77, 77};
    for (uint32_t i = 0; i < 2000; ++i) ids.push_back(i); // more jobs than the in-flight limit
    std::vector<uint8_t> out(ids.size() * rb);
    for (int turn = 0; turn < 3; ++turn) {
        auto t = r.issue(ids.data(), ids.size(), out.data()); CHECK(r.collect(t, err)); r.flush_disk_cache();
        for (size_t i = 0; i < ids.size(); ++i) for (uint32_t j = 0; j < rb; ++j)
            CHECK(out[i * rb + j] == (ids[i] < rows ? contents[offset + (uint64_t) ids[i] * rb + j] : 0));
    }
    CHECK(r.snapshot().disk_hits > 0 && r.snapshot().cache_hits == 0);
    // Truncating the cache after warming it triggers a source fallback, with identical rows.
    fs::resize_file(chunk(a), 0);
    auto t = r.issue(ids.data(), ids.size(), out.data()); CHECK(r.collect(t, err)); r.flush_disk_cache();
    for (size_t i = 0; i < ids.size(); ++i) for (uint32_t j = 0; j < rb; ++j)
        CHECK(out[i * rb + j] == (ids[i] < rows ? contents[offset + (uint64_t) ids[i] * rb + j] : 0));
    // Multiple outstanding tickets, collected out of issue order.
    std::vector<uint8_t> other(out.size());
    auto t1 = r.issue(ids.data(), ids.size(), out.data());
    auto t2 = r.issue(ids.data(), ids.size(), other.data());
    CHECK(r.collect(t2, err)); CHECK(r.collect(t1, err)); CHECK(other == out);
    r.flush_disk_cache();
    r.close();
    // Close while reads are outstanding must drain both source and cache workers safely.
    CHECK(r.open(source.string(), offset, rows, 16, 0, err, threaded, rb, {{a.string(), 2}, {b.string(), 2}}));
    r.issue(ids.data(), ids.size(), out.data()); r.close();
    // Invalid optional storage degrades to a source read, not an inference failure.
    CHECK(r.open(source.string(), offset, rows, 16, 0, err, threaded, rb, {{source.string(), 2}}));
    t = r.issue(ids.data(), ids.size(), out.data()); CHECK(r.collect(t, err)); r.flush_disk_cache(); r.close();
    fs::remove(source);
}
int main(int argc, char** argv) {
    const auto base = temp(fs::temp_directory_path());
    const auto a = temp(argc > 1 ? fs::path(argv[1]) : base);
    const auto b = temp(argc > 2 ? fs::path(argv[2]) : base);
    core(a,b);
    for (uint32_t rb : {90u,160u,320u}) for (bool threaded : {false,true}) reader(base,a,b,rb,threaded);
    fs::remove_all(a); fs::remove_all(b); fs::remove_all(base);
    std::cout << "PLE disk cache tests passed: two roots, bounded eviction, corruption, failure, crash cleanup, exact rows (90/160/320 B, worker/caller).\n";
}
