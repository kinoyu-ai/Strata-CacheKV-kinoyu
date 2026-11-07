# Disposable disk conversation cache fork

## What this fork changes

Based on Strata v0.2.10 / engine 0.1.56, this fork parks long outgoing conversations in temporary files when
another conversation takes over. It stores each GPU stage in its own session-format file and registers the
group only after every stage is written. All files, stage geometry, tokens, images and checkpoints are validated
before any GPU is overwritten. K/V is read and written in bounded blocks; running state and checkpoints still
need RAM. A storage or validation failure falls back to prompt processing. A failed transfer after device writes
begin stops the engine rather than continuing from partial state.

The disk tier supports Linux, ordinary sequential serving and layer splits, with MTP on its owning stage.
`--batch` and `--peer-device` are not supported. CUDA is the only backend tested for this fork. HIP, SYCL,
Metal and Windows support for the new tier are not claimed. Existing RAM cache metrics do not include disk entries.

There is no change to the public SAVE/RESTORE commands; they still reject layer splits.

## Build this source

Install the compilers and CUDA toolkit required by [the source build guide](SOURCE_INSTALL.md).
After checking out this fork's `disk-conversation-cache` branch, build its engine explicitly:

```sh
cmake -S . -B build-disk-cache -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build-disk-cache --target strata -j2
```

For a V100/Volta build like the validation machine, use CUDA **12.x** (12.9 was tested) and replace
`-DCMAKE_CUDA_ARCHITECTURES=native` with
`-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`.
Upstream treats these older GPUs as community builds; CUDA 13 cannot compile sm_70.

Point your existing Strata server configuration's `exe` to the resulting engine. Keep your own model paths,
tokenizer, GPU layout and authentication. Models, packs, credentials and machine-specific configuration are not
included. The upstream setup workflow may download an upstream binary unless a source build is explicitly requested;
see `./setup.sh --build --no-start` in the source guide.

## Enable explicitly

Add these pairs to the engine `args` of your existing model configuration:

```json
[
  "--conversation-cache-mib", "8192",
  "--conversation-cache-slots", "64",
  "--conversation-cache-min-tokens", "0",
  "--conversation-cache-tier-order", "ram-first",
  "--conversation-cache-dir", "/mnt/fast-cache/strata-kv",
  "--conversation-cache-disk-mib", "12288",
  "--conversation-cache-disk-min-tokens", "60001",
  "--conversation-cache-disk-min-free-mib", "512"
]
```

This is an argument excerpt, not a complete model configuration. `/mnt/fast-cache/strata-kv` is an example: replace
it with a dedicated directory on your chosen mounted storage, owned by the service user without group/other write
access. Parent directories must exist and be traversable; the engine can create the leaf directory. Confirm the
storage is mounted before starting. No device is formatted or selected automatically. Restart the engine after
changing its arguments. An Optane device is optional; it has not been tested for this fork.

The example uses up to 8192 MiB and 64 parked conversations in RAM. `ram-first` is the new default placement:
use RAM first, then spill the least recently used unpinned/unguarded RAM entry on a byte/slot/physical-memory eviction.
A state too large or unable to enter RAM is streamed straight from the active session to disk instead.
`disk-first` selects the earlier behavior and reproduces the placement used for the original benchmark below.
Short conversations can remain in RAM (`--conversation-cache-min-tokens 0`), while disk requires 60001 tokens.
RAM spills hand ownership of the existing buffers to the disk writer; no complete K/V duplicate is created.
Every GPU stage is registered only after all files succeed. Explicit/poison flushes never spill.

The RAM budget is for parked conversation images, not the entire process. The existing 2560 MiB free-RAM floor
remains enforced. Restores may spill/release older RAM entries to admit their state, while protecting the incoming
disk group and any borrowed RAM donor. The disk tier has its own 512 MiB free-space floor; ordinary session SAVE
keeps its separate existing setting.

The default capacity is 12288 MiB (12 GiB) and the default minimum captured conversation length is 60001 tokens.
The minimum counts captured state, including processed generated tokens. Last-used entries are evicted first.
The incoming entry is protected while parking the outgoing one. An entry larger than the budget is skipped.
The RAM and disk minimum lengths are independent. Set the RAM minimum to 60001 to exclude short RAM entries too.
Set `--conversation-cache-mib 0` for disk-only parking; remove the disk-directory option for RAM-only parking.

## Disable and clean up

Remove `--conversation-cache-dir` and its value, or set `--conversation-cache-disk-mib 0`, then restart.
To restore the original auto-sized RAM policy, remove the explicit RAM budget, slot count and minimum-token options.
Without a directory, no disk tier is enabled. No disk cache is restored across restarts.

QUIT/EOF removes the process's run directory. A killed process can leave files until the next start; that start
acquires the directory lock and removes stale runs. A second live engine cannot use or purge the same root.
If your supervisor kills the engine, it can clean up stale `run-*` folders after acquiring the same `.lock`
exclusively and confirming no engine owns it. Never delete another engine's live files.
The files are disposable and are not flushed for crash durability. Normal prompt checkpoints are unchanged.

Logs containing `disk cache: parked`, `disk cache: read`, `disk cache: restored`, and removal/refusal reasons
show activity. A directory, capacity and minimum line appears when the tier starts; startup refusals explicitly
report that disk caching is disabled.

## Validation actually performed

Passed on Linux with two V100 GPUs, CUDA 12.9, sm_70, Qwen3.8-Flash-Next IQ3_S, `k8v4`, MTP,
262144-token context, parallelism 1 and layer split 24:

- Nine C++ tests: cache policy, split failures, memory admission, session file I/O, save reclamation,
  disposable disk lifecycle, GPU snapshot byte equality, validation, and transfer failure handling.
- `python -m unittest serve.test_server`: 359 existing API tests passed.
- A 60000/60001 captured-token boundary, bounded capacity/LRU, failed second-stage saves, stale-run cleanup,
  active-engine locking, corrupt files and RAM refusal were checked in unit tests.
- A file with more than 65 MiB of K/V was read with an admission callback allowing less than 40 MiB of read
  allocations, and no full K/V host allocation. GPU snapshot tests include hybrid K8/V4 and MTP ring equality.

To run the C++ checks after building their targets:

```sh
cmake --build build-disk-cache -j2 --target \
  conversation_cache_test conversation_split_failure_test conversation_memory_test \
  conversation_file_test session_save_reclaim_test conversation_disk_test \
  conversation_snapshot_test conversation_validation_test conversation_transfer_test
ctest --test-dir build-disk-cache --output-on-failure \
  -R '^(conversation_(cache|split_failure|memory|file|disk|snapshot|validation|transfer)_test|session_save_reclaim_test)$'
python -m unittest serve.test_server
```

GPU tests require available GPU resources. The transfer test is conditional in CMake; use a supported Linux/GNU CUDA
build for the full set. Do not run model tests on GPUs occupied by another production engine.

Observed synthetic workload, one run per case, 62074 input tokens, temperature 0, top-k 1, seed 42, thinking off:

| Case | Reused tokens | Prefill time |
| --- | ---: | ---: |
| Original engine, cold | 0 | 36.19 s |
| Modified engine, feature disabled, cold | 0 | 36.04 s |
| Disk enabled, cold | 0 | 36.55 s |
| Long A → short request → A | 62067 | 2.53 s |
| Long A → long B → A | 62067 | 4.15 s |
| Corrupted second-stage file → A recomputed | 0 | 37.69 s |

Times are the engine-reported `timings.prompt_ms` divided by 1000. They include input processing such as
cache parking/restoration and exclude answer generation (decode).

Every A answer matched the original engine. The A snapshot was 1006707056 bytes across two files; parking took
1.37–1.75 s. Normal exit removed the files. The corruption case rejected a payload checksum mismatch before restoring
and recomputed the answer. The tests kept both GPUs at 150 W, cooled below 50 C between cases and used a 75 C test stop;
the highest sampled temperature was 60 C. Host MemAvailable remained above the existing 2560 MiB floor.

Limits: no statistical speedup claim, no OS page-cache flush, no full 12 GiB physical fill test, no Optane hardware test,
and no HIP/SYCL/Metal/Windows build validation. Capacity/LRU tests use small synthetic files. Performance on other models,
conversation shapes and storage devices has not been measured. The feature stays opt-in.

The new RAM-first integration was checked in the local 0.2.11 CUDA port on two V100s with MTP. A test-only
one-entry RAM limit and disk minimum of one token forced A (2652 tokens) -> short S -> short B -> A to spill
and restore 2645 tokens; the response matched the original engine on the same input. This KV test used a temporary
NVMe directory, not the Optane drive, and is not an 8 GiB saturation or production-threshold performance benchmark.
