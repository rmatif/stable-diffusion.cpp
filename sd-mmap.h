#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct sd_file;
struct sd_mmap;
struct sd_mlock;

using sd_files  = std::vector<std::unique_ptr<sd_file>>;
using sd_mmaps  = std::vector<std::unique_ptr<sd_mmap>>;
using sd_mlocks = std::vector<std::unique_ptr<sd_mlock>>;

struct sd_file {
    sd_file(const char * fname, const char * mode);
    ~sd_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const;

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len) const;
    uint32_t read_u32() const;

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct sd_mmap {
    sd_mmap(const sd_mmap &) = delete;
    sd_mmap(struct sd_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~sd_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct sd_mlock {
    sd_mlock();
    ~sd_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t sd_path_max();