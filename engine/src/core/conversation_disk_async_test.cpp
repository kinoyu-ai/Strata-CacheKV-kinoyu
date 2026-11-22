#include "strata/core/conversation_disk_async.hpp"
#include <atomic>
#include <cassert>
#include <fstream>
#include <unistd.h>
using namespace strata::core;
namespace fs = std::filesystem;
static void check(bool value) { if (!value) std::abort(); }
static SavedConversation sample(int token) {
    SavedConversation s; s.live.ids.assign(32,token); s.live.gdn.assign(128,token); s.layer_hi=24;
    ConversationKv kv; kv.format=4; kv.cells=32; kv.heads=1; kv.head_dim=32; kv.page_size=32;
    kv.k.resize(1024,token); kv.v.resize(512,token+1); s.kv.push_back(std::move(kv));
    auto part=s; part.layer_lo=24; part.layer_hi=48; s.stage_images.push_back(std::move(part));
    return s;
}
int main() {
    auto root=fs::temp_directory_path()/("strata-async-test-"+std::to_string(getpid()));
    fs::create_directory(root); fs::permissions(root,fs::perms::owner_all);
    const SessionFileIdentity id{123,456}; std::string error;
    {
        ConversationDiskCache cache; check(cache.open(root.string(),4u<<20,0,error,true,99,32768));
        ConversationDiskAsync async(cache);
        std::promise<void> started, release; auto gate=release.get_future().share(); std::atomic<bool> once=false;
        SessionWriteOptions options; options.durable=true;
        options.progress=[&](uint64_t,uint64_t) { if (!once.exchange(true)) { started.set_value(); gate.wait(); } };
        auto original=sample(7);
        check(async.submit(std::move(original),id,1u<<20,options,false,error));
        check(started.get_future().wait_for(std::chrono::seconds(5))==std::future_status::ready);
        // A deliberately stopped disk worker must not hold cache locks, publish a partial group,
        // or prevent the caller from doing the next generation's host work.
        check(cache.best(std::vector<int32_t>(33,7),{},true).tokens==0);
        async.poll(); check(async.busy());
        check(!async.submit(sample(9),id,1u<<20,{},false,error));
        auto next=sample(99); check(next.live.ids[0]==99);
        release.set_value(); async.poll(true); check(!async.busy());
        auto m=cache.best(std::vector<int32_t>(33,7),{},true); auto e=cache.find(m.serial);
        check(e&&e->files.size()==2);
        for(size_t k=0;k<2;++k) { SavedConversation s; size_t n=0;
            check(session_file_read(e->files[k],id,s,n,error));
            auto want=sample(7); auto& part=k?want.stage_images[0]:want;
            check(s.live.ids==part.live.ids && s.kv[0].k==part.kv[0].k && s.kv[0].v==part.kv[0].v);
        }
        // Disk-full rejection never publishes half a group.
        options={}; options.min_free_bytes=UINT64_MAX;
        check(async.submit(sample(8),id,1u<<20,options,false,error)); async.poll(true);
        check(cache.size()==1);
        // Failure after the first GPU file was flushed must still publish neither stage.
        auto flushes=std::make_shared<std::atomic<int>>(0);
        options={}; options.fault=[flushes](const char* step) {
            return std::string(step)=="file_flush" && ++*flushes==2 ? EIO : 0;
        };
        check(async.submit(sample(10),id,1u<<20,options,false,error)); async.poll(true);
        check(cache.size()==1 && cache.best(std::vector<int32_t>(33,10),{},true).tokens==0);
    }
    fs::create_directory(root/"persistent"/"async-pending");
    std::ofstream(root/"persistent"/"async-pending"/"0.bin") << "interrupted transfer";
    // Committed data survives restart; unfinished transfer files do not.
    {
        ConversationDiskCache cache; check(cache.open(root.string(),4u<<20,0,error,true,99,32768));
        check(cache.size()==1 && cache.best(std::vector<int32_t>(33,7),{},true).tokens==32);
        check(!fs::exists(cache.staging_path()));
    }
    fs::remove_all(root); std::puts("async disk cache: overlap, ownership, split-stage restore, restart and failure passed");
}
