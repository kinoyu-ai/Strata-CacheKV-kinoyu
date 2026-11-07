# Optional Ngram (PLE) disk page cache

This opt-in Linux tier caches immutable source-table pages after a miss in the existing RAM row cache.
It is independent of the conversation KV cache. The original model files remain authoritative and must stay available.
It is off by default and does not preserve cache contents across process restarts.

Add the following argument excerpt to the configuration for an engine built from this fork:

```json
[
  "--ple-io", "direct",
  "--ple-row-cache", "1048576",
  "--ple-disk-cache", "/mnt/cache-a/strata-ngram-cache", "12288",
  "--ple-disk-cache", "/mnt/cache-b/strata-ngram-cache", "12288"
]
```

Repeat `--ple-disk-cache PATH MIB` for each storage directory. Each directory has its own byte budget; the example
uses 12 GiB on each drive, 24 GiB total. Page numbers select a directory, with bounded hash replacement within it.
One directory also works. Zero MiB disables that directory. The RAM row count and conversation-cache budgets are independent.
The total table need not fit. No RAID or model relocation is required.

Pre-create dedicated directories owned by the service user, without group/other write access, on the mounted drives.
Directories are deliberately not created automatically, to avoid silently caching onto the system filesystem when a
removable drive is absent. A root lock excludes a second engine using the same directory. `--ple-io mmap|ram` with
this option is rejected. Unsupported optional storage falls back to source reads. The new disk tier is Linux-only;
other backends/platforms have not been validated.

Data is stored with direct I/O in files no larger than 256 MiB, including on supported FAT32 filesystems.
Cache reads use up to 16 workers. Source-read completions queue background fills rather than waiting for writes.
The fill queue is bounded at 16 MiB; excess optional fills are skipped. Only tags, checksums and small buffers live in RAM: two 12 GiB roots need about 144 MiB of index metadata, plus file
bookkeeping. A page is published only after its full write. Checksum failures miss and refill from the original
source; read/write errors or insufficient space disable the tier while source reads continue. A process-specific
run directory is deleted on close. Stale runs are deleted on the next exclusive open and are never reused for another
model. The startup log identifies roots and budgets; periodic/close logs report hits, misses, writes, corrupt pages,
failures and metadata usage.

To disable, remove each option and its following path and capacity, then restart. Other caches need no changes.
When sharing physical storage with the KV disk tier, account for the sum of the budgets and filesystem overhead.
Cache fills add writes; USB latency or slow cache storage can make inference slower. No speed improvement is claimed.
The README's KV prefill benchmark does not measure this Ngram tier.

## CPU and storage validation

```sh
g++ -std=c++20 -O2 -pthread -I include \
  src/ngram/ple_disk_cache_test.cpp src/ngram/ple_reader.cpp \
  src/platform/direct_file.cpp -o /tmp/ple-disk-cache-test
/tmp/ple-disk-cache-test
# Optional: two existing writable roots; only newly created test directories are removed.
/tmp/ple-disk-cache-test /mnt/cache-a/strata-ngram-cache /mnt/cache-b/strata-ngram-cache
```

Tests exercise two roots, byte limits/replacement, corruption, failed cache reads, lock exclusion, graceful/crash
cleanup, exact source bytes with 90/160/320-byte rows including page straddles and EOF, worker/caller submission,
and optional-cache open failures. The same rows must match with the cache off, on a cold fill and on cache hits.

## Validation performed

Eight public v0.2.10 CPU tests (including existing KV tests), the existing PLE selftest across eight formats,
ASan/UBSan, and a non-PIE TSan build passed. A regular PIE TSan run could not start because of the host's
address-space mapping; the non-PIE run completed. The same cache tests passed on two real USB-attached ext4 roots.
The local CUDA engine was built from a 0.2.11 port of these changes and checked on two V100s with Qwen3.8-Flash-Next
Unsloth v3.1 UD-IQ3_XXS, k8v4 and MTP. The public v0.2.10 engine translation unit also passed a syntax check with the
CUDA build flags. Other GPU backends and a real 24 GiB saturation run were not tested.

A raw I/O probe of 4096 random 90-byte rows with the RAM row cache disabled took about 37–40 ms from the original
NVMe and 66–67 ms after disk-cache hits on these USB drives, with identical bytes. It did not show a speedup.
Each root used 512 MiB; the measurement ends at read completion and excludes waiting for background fills.
This was a reference probe concurrent with CPU compilation, not a prefill benchmark or an end-to-end performance claim.


The v0.2.12 update passed the same eight host tests, PLE selftest and real two-Optane tests, plus five GPU state tests and 376 API tests. The v0.2.12 CUDA engine also restored a 62k-token KV group from a dedicated third Optane with matching answers. See [version-specific validation](DISK_KV_CACHE_FORK.md#v0212-upgrade-validation-2026-11-08).
