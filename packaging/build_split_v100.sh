#!/usr/bin/env bash
# Build an independent, experimental MTP split engine for two V100s. Does not edit serving configuration.
set -euo pipefail
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
B=${B:?Set B to a separate build directory}
CUDA=${CUDA:?Set CUDA to a CUDA 12.x toolkit root}
GG=${GG:?Set GG to the pinned llama.cpp checkout}
JOBS=${JOBS:-4}
cmake -S "$SRC/packaging/split-v100" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_GGML_DIR="$GG" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build "$B" --target strata-split-2 split_moe_tile_test split_card_check_test split_expert_residency_test -j "$JOBS"
exe="$B/split/strata-split-2"
if nm -D --undefined-only "$exe" | grep -qi peer; then
  echo "V100 split build unexpectedly references a CUDA peer API" >&2
  exit 1
fi
"$B/split/split_card_check_test"
"$B/split/split_expert_residency_test"
printf '%s\n' "$exe"
printf 'Run with LD_LIBRARY_PATH=%s/lib64:${LD_LIBRARY_PATH:-} (model/VRAM/RAM fit is still required).\n' "$CUDA"
