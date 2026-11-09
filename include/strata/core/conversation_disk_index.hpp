#pragma once
// Small durable index only. GPU session files remain the authority and are fully validated before restore.
#include "strata/core/conversation_cache.hpp"
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace strata::core::disk_index {
constexpr size_t max_bytes = 64u << 20;
inline uint64_t hash(const std::vector<uint8_t>& b, size_t n) {
    uint64_t h=1469598103934665603ull;
    for(size_t i=0;i<n;++i) h=(h^b[i])*1099511628211ull;
    return h;
}
struct Codec {
    std::vector<uint8_t> b; size_t at=0;
    void put(uint64_t v, size_t n=8) { for(size_t i=0;i<n;++i) b.push_back((uint8_t)(v>>(i*8))); }
    uint64_t get(size_t n=8) {
        if(n>b.size()-at) throw std::runtime_error("truncated disk index");
        uint64_t v=0;for(size_t i=0;i<n;++i)v|=(uint64_t)b[at++]<<(8*i);return v;
    }
    void checkpoint(const ConversationCheckpoint& c) {
        put(c.ids.size());for(int32_t t:c.ids) put((uint32_t)t,4);
        put(c.imgs.size());for(const auto& x:c.imgs){put((uint64_t)x.start);put(x.hash);}
    }
    void checkpoint(ConversationCheckpoint& c, uint64_t max_tokens) {
        uint64_t n=get();if(n>max_tokens||n>(b.size()-at)/4)throw std::runtime_error("disk index token count");
        c.ids.resize((size_t)n);for(auto& t:c.ids)t=(int32_t)(uint32_t)get(4);
        n=get();if(n>max_tokens||n>(b.size()-at)/16)throw std::runtime_error("disk index image count");
        c.imgs.resize((size_t)n);for(auto& x:c.imgs){x.start=(int64_t)get();x.hash=get();}
    }
};
#if defined(__linux__)
inline void sync_dir(const std::filesystem::path& path) {
    int fd=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0)throw std::runtime_error("open disk index directory");
    int r=::fsync(fd);::close(fd);if(r)throw std::runtime_error("flush disk index directory");
}
inline void write(const std::filesystem::path& path,const std::vector<uint8_t>& b) {
    if(b.size()>max_bytes)throw std::runtime_error("disk index too large");
    std::string pattern=(path.parent_path()/".index-XXXXXX").string();
    std::vector<char> tmp(pattern.begin(),pattern.end());tmp.push_back(0);
    int fd=::mkstemp(tmp.data());if(fd<0)throw std::runtime_error("create disk index");
    bool renamed=false;
    try {
        if(::fchmod(fd,0600))throw std::runtime_error("disk index permissions");
        size_t at=0;while(at<b.size()) {
            ssize_t n=::write(fd,b.data()+at,b.size()-at);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("write disk index");at+=(size_t)n;
        }
        if(::fsync(fd))throw std::runtime_error("flush disk index");
        ::close(fd);fd=-1;
        if(::rename(tmp.data(),path.c_str()))throw std::runtime_error("publish disk index");
        renamed=true;sync_dir(path.parent_path());
    } catch(...) {if(fd>=0)::close(fd);if(!renamed)::unlink(tmp.data());throw;}
}
inline Codec read(const std::filesystem::path& path) {
    int fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    if(fd<0)throw std::runtime_error("open disk index");
    try {
        struct stat st{};
        if(::fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_nlink!=1||st.st_uid!=::getuid()||
           (st.st_mode&0022)||st.st_size<80||(uint64_t)st.st_size>max_bytes)
            throw std::runtime_error("disk index file bounds/type/owner");
        Codec c;c.b.resize((size_t)st.st_size);size_t at=0;
        while(at<c.b.size()) {
            ssize_t n=::read(fd,c.b.data()+at,c.b.size()-at);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("read disk index");at+=(size_t)n;
        }
        ::close(fd);return c;
    } catch(...) {::close(fd);throw;}
}
#else
inline void sync_dir(const std::filesystem::path&) { throw std::runtime_error("persistent disk index requires Linux"); }
#endif
} // namespace strata::core::disk_index
