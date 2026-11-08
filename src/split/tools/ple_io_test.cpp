// The optional split PLE backend must return the original bytes, including page-straddling and final rows.
#include "ple_io.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <unistd.h>
static void require(bool ok) { if (!ok) throw std::runtime_error("PLE I/O test failed"); }
int main() {
    char tmp[]="/tmp/strata-split-ple-XXXXXX";
    const char* dir=mkdtemp(tmp); if (!dir) return 1;
    try {
        const std::string path=std::string(dir)+"/table";
        std::filesystem::create_directory(std::string(dir)+"/cache-a");
        std::filesystem::create_directory(std::string(dir)+"/cache-b");
        for (const auto* name : {"cache-a","cache-b"})
            std::filesystem::permissions(std::string(dir)+"/"+name,std::filesystem::perms::owner_all);
        for (uint32_t rb : {90u,170u}) {
            const uint32_t nr=4097; const uint64_t offset=192;
            std::vector<uint8_t> bytes(offset+(uint64_t)nr*rb);
            for(size_t i=0;i<bytes.size();++i) bytes[i]=(uint8_t)((i*97)^(i>>9));
            {std::ofstream f(path,std::ios::binary); f.write((const char*)bytes.data(),bytes.size()); require((bool)f);}
            for (uint64_t cache : {0u,128u}) {
                fnx::ple_io_configure(true,cache,{{std::string(dir)+"/cache-a",1},{std::string(dir)+"/cache-b",1}});
                fnx::ple_io_open(path,offset,nr,rb);
                std::vector<uint32_t> rows{0,1,45,46,4095,4096,45};
                for(uint32_t i=0;i<2048;++i) rows.push_back((i*991u)%nr);
                std::vector<uint8_t> out(rows.size()*rb);
                for(int pass=0;pass<2;++pass) {
                    fnx::ple_io_read(rows.data(),rows.size(),out.data());
                    for(size_t j=0;j<rows.size();++j)
                        for(size_t b=0;b<rb;++b) require(out[j*rb+b]==bytes[offset+(uint64_t)rows[j]*rb+b]);
                }
                fnx::ple_io_close();
                bool threw=false;try{fnx::ple_io_read(rows.data(),rows.size(),out.data());}catch(const std::exception&){threw=true;}require(threw);
            }
        }
        fnx::ple_io_configure(false,0,{});require(!fnx::ple_io_direct());
        bool threw=false;try{fnx::ple_io_configure(false,0,{{dir,1}});}catch(const std::exception&){threw=true;}require(threw);
        fnx::ple_io_configure(true,0,{});
        threw=false;try{fnx::ple_io_open(path+"-missing",192,1,90);}catch(const std::exception&){threw=true;}require(threw);
        threw=false;try{fnx::ple_io_open(path,192,1ull<<32,90);}catch(const std::exception&){threw=true;}require(threw);
        fnx::ple_io_close();std::filesystem::remove_all(dir);
        std::puts("split PLE I/O: PASS");return 0;
    } catch(const std::exception& e) {fnx::ple_io_close();std::filesystem::remove_all(dir);std::fprintf(stderr,"%s\n",e.what());return 1;}
}
