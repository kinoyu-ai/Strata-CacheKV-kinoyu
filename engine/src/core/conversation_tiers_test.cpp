#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_file.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
using namespace strata::core;
namespace fs = std::filesystem;
static void check(bool b,const char* msg) { if(!b){std::fprintf(stderr,"FAIL: %s\n",msg);std::exit(1);} }
static SavedConversation sample(int token) {
    SavedConversation s;
    s.live.ids.assign(60001,token); s.live.gdn.assign(4096,(uint8_t)token);
    s.checkpoints.push_back(s.live); s.checkpoints.back().ids.resize(60000);
    s.layer_hi=24;
    ConversationKv kv;kv.format=4;kv.cells=60001;kv.heads=2;kv.head_dim=256;kv.page_size=64;
    kv.k.resize(8192,(uint8_t)token);kv.v.resize(4096,(uint8_t)(token+1));
    s.kv.push_back(std::move(kv));
    auto part=s;part.layer_lo=24;part.layer_hi=48;
    // Stand-in for an owning stage's MTP K/V; the host format must preserve both layers.
    part.kv.push_back(part.kv.front());s.stage_images.push_back(std::move(part));
    return s;
}
int main(int argc,char** argv) {
    auto parent=argc>1?fs::path(argv[1]):fs::temp_directory_path();
    auto root=parent/("tiers-test-"+std::to_string(getpid()));fs::create_directory(root);fs::permissions(root,fs::perms::owner_all);
    std::string error;const SessionFileIdentity id{123,456};
    {
        ConversationDiskCache disk;check(disk.open(root.string(),16u<<20,60001,error),"open disk");
        ConversationCache ram(8u<<20,1);int attempts=0;bool fail=false;
        ram.set_eviction_sink([&](SavedConversation&& image){
            ++attempts;
            const uint64_t estimate=image.bytes()+(2u<<20);
            const bool ok=disk.store(image.live.ids,image.live.imgs,image.checkpoints,image.cvec,2,estimate,
                [&](const std::string& path,size_t stage,size_t& bytes,std::string& why){
                    if(fail&&stage==1){why="injected second-stage failure";return false;}
                    SessionWriteOptions opts;opts.durable=false;opts.min_free_bytes=0;
                    return session_file_write(path,stage?image.stage_images[stage-1]:image,id,bytes,why,opts);
                },error);
            check(ok!=fail,"all stages saved or failure stays a miss");
        });
        check(ram.put(sample(1)),"A parked in RAM");check(attempts==0&&disk.size()==0,"disk untouched while RAM fits");
        check(ram.put(sample(2)),"B enters RAM");check(attempts==1&&disk.size()==1,"A spilled when RAM slot filled");
        auto ids=std::vector<int32_t>(60001,1);ids.push_back(9);auto match=disk.best(ids,{},true);
        check(match.tokens==60001,"A matches disk");const auto* entry=disk.find(match.serial);check(entry&&entry->files.size()==2,"two stage files");
        SavedConversation loaded;
        for(size_t k=0;k<2;++k){
            SavedConversation part;size_t bytes=0;SessionReadLimits limits;limits.stream_kv=true;
            check(session_file_read(entry->files[k],id,part,bytes,error,limits),"stream spilled stage");
            auto expected=sample(1);const auto& want=k?expected.stage_images[0]:expected;
            check(part.live.ids==want.live.ids&&part.live.gdn==want.live.gdn&&part.kv.size()==want.kv.size(),"running state and MTP layer count");
            for(size_t i=0;i<part.kv.size();++i)check(part.kv[i].k==want.kv[i].k&&part.kv[i].v==want.kv[i].v,"exact KV bytes");
            if(k)loaded.stage_images.push_back(std::move(part));else loaded=std::move(part);
        }
        // Promotion to RAM spills B. The streamed backing is kept valid while its directory is replaced/evicted.
        check(ram.put(std::move(loaded)),"A promoted to RAM");check(attempts==2&&disk.size()==2,"B spilled on promotion");
        ram.clear();check(attempts==2,"clear does not spill");disk.clear();
        check(ram.put(sample(3)),"before failing spill");fail=true;
        check(ram.put(sample(4)),"RAM remains usable after storage failure");check(disk.size()==0,"partial stage group never registered");
    }
    fs::remove_all(root);std::puts("RAM-first KV tiers: capacity spill, two stages/MTP bytes, streamed promotion, failed group and clear passed.");
}
