#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_file.hpp"
#include <fstream>
#include <sys/wait.h>
using namespace strata::core;
namespace fs=std::filesystem;
static void check(bool b,const char* m){if(!b){std::fprintf(stderr,"FAIL %s\n",m);std::exit(1);}}
static SavedConversation sample(int token) {
    SavedConversation s;s.live.ids.assign(32,token);s.live.gdn.assign(128,(uint8_t)token);
    s.checkpoints.push_back(s.live);s.checkpoints.back().ids.resize(31);s.layer_hi=24;
    ConversationKv kv;kv.format=4;kv.cells=32;kv.heads=1;kv.head_dim=32;kv.page_size=32;
    kv.k.resize(1024,token);kv.v.resize(512,token+1);s.kv.push_back(std::move(kv));
    auto part=s;part.layer_lo=24;part.layer_hi=48;part.kv.push_back(part.kv[0]);s.stage_images.push_back(std::move(part));return s;
}
int main(){
 const auto root=fs::temp_directory_path()/("strata-persistent-test-"+std::to_string(getpid()));
 fs::create_directory(root);fs::permissions(root,fs::perms::owner_all);
 const SessionFileIdentity identity{123,456};std::string error;uint64_t serial=0;const uint64_t key=99;
 auto save=[&](ConversationDiskCache& c,int token,bool mirror=true){
  auto s=sample(token);
  return c.store(s.live.ids,s.live.imgs,s.checkpoints,true,2,1u<<20,
   [&](const std::string& file,size_t k,size_t& n,std::string& e){return session_file_write(file,k?s.stage_images[0]:s,identity,n,e);},error,mirror);
 };
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,60001,error,true,key,32768),"open persistent");
  check(!save(c,1,false),"short disk-only excluded");check(save(c,1),"short RAM mirror saved despite spill minimum");
  serial=c.best(std::vector<int32_t>(33,1),{},true).serial;check(serial!=0,"mirror match");
  auto before=c.bytes();check(save(c,1)&&c.bytes()==before&&c.size()==1,"unchanged snapshot not rewritten");
  ConversationDiskCache other;check(!other.open(root.string(),4u<<20,0,error,true,key),"exclusive lock");
 }
 check(fs::exists(root/"persistent"/std::to_string(serial)/"index.bin"),"survives clean close");
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,60001,error,true,key,32768)&&c.size()==1,"index after restart");
  auto match=c.best(std::vector<int32_t>(33,1),{},true);auto e=c.find(match.serial);check(e&&e->files.size()==2,"both GPUs indexed");
  for(size_t k=0;k<2;++k){SavedConversation loaded;size_t bytes=0;
   check(session_file_read(e->files[k],identity,loaded,bytes,error),"validate payload after restart");
   auto want=sample(1);auto& part=k?want.stage_images[0]:want;
   check(loaded.live.ids==part.live.ids&&loaded.live.gdn==part.live.gdn&&loaded.kv.size()==part.kv.size(),"state and MTP restored");
   for(size_t i=0;i<loaded.kv.size();++i)check(loaded.kv[i].k==part.kv[i].k&&loaded.kv[i].v==part.kv[i].v,"KV bytes exact");
  }
  c.touch(match.serial);check(save(c,2),"second entry");
 }
 // Missing stage is not registered even though its index exists.
 fs::remove(root/"persistent"/std::to_string(serial)/"stage-1.bin");
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,0,error,true,key)&&c.size()==1,"missing GPU rejected");c.clear();
 }
 // Simulate death while writing the second GPU. No index publishes a partial group.
 auto child=fork();if(!child){ConversationDiskCache c;std::string e;if(!c.open(root.string(),4u<<20,0,e,true,key))_exit(2);
  c.store({1,2},{},{},true,2,1u<<20,[&](const std::string& path,size_t k,size_t& n,std::string&){if(k)_exit(0);std::ofstream(path)<<"partial";n=7;return true;},e);_exit(3);}
 int status=0;waitpid(child,&status,0);check(status==0,"partial child exit");
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,0,error,true,key)&&c.size()==0,"orphan cleaned");
 }
 child=fork();if(!child){ConversationDiskCache c;if(!c.open(root.string(),4u<<20,0,error,true,key)||!save(c,3))_exit(2);_exit(0);}
 waitpid(child,&status,0);check(status==0,"committed child exit");
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,0,error,true,key)&&c.size()==1,"committed group survives abrupt exit");
 }
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,0,error,true,key+1)&&c.size()==0,"different identity discarded");check(save(c,4),"new identity");
 }
 // A corrupted or truncated index is only a cache miss.
 for(const auto& dir:fs::directory_iterator(root/"persistent"))std::ofstream(dir.path()/"index.bin",std::ios::trunc)<<"bad";
 {
  ConversationDiskCache c;check(c.open(root.string(),4u<<20,0,error,true,key+1)&&c.size()==0,"bad index discarded");
 }
 // LRU counts metadata bytes and honours touches across restarts (small, synthetic stage files).
 auto tiny=[&](ConversationDiskCache& c,int token){return c.store({token},{},{},true,2,256,
  [](const std::string& path,size_t,size_t& n,std::string&){std::ofstream(path)<<std::string(16,'x');fs::permissions(path,fs::perms::owner_read|fs::perms::owner_write);n=16;return true;},error,true);};
 uint64_t keep=0;
 {ConversationDiskCache c;check(c.open(root.string(),700,0,error,true,key),"tiny open");
  check(tiny(c,1)&&tiny(c,2)&&tiny(c,3),"tiny entries");keep=c.best(std::vector<int32_t>{1,9},{},true).serial;c.touch(keep);}
 {ConversationDiskCache c;check(c.open(root.string(),700,0,error,true,key),"tiny restart");
  check(c.find(keep)&&c.find(keep)->live.ids[0]==1,"old entry indexed before new store");
  check(tiny(c,4)&&c.bytes()<=700,"capacity bounded including index");
  check(c.find(keep)&&c.find(keep)->live.ids[0]==1,"LRU touch survives restart");c.clear();}
 fs::remove_all(root);std::puts("persistent KV index: restart, mirrors, two stages/MTP bytes, crash/partial, identity, corruption, LRU PASS");
}
