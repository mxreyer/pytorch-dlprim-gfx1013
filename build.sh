#!/usr/bin/env bash
# Rebuild pytorch_ocl (pytorch_dlprim) with the four BC-250 patches applied:
#   custom_reduce.patch   - portable work-group reduction; rusticl has no
#                           work_group_reduce_* built-ins (correctness)
#   winograd_ksplit.patch - fill all CUs in Winograd backward-filter, replace
#                           both backward kernels' emulated fp32 atomics with
#                           plane writes + a reduce, coalesce four access
#                           patterns, prefetch, opt-in fp16 inner loop
#                           (909 -> 2,055 img/s; 2,987 with DLPRIM_CONV_FP16=1)
#   gelu_mad.patch        - fma() is software-emulated on rusticl < Mesa 26.2;
#                           use mad() (redundant, harmless on 26.2)
#   pytorch_ocl_half_fixes.patch - reject non-float32 tensors in convolution
#                           (they silently produced NaN); dtype-correct
#                           hardtanh/relu6/clamp; contiguous grad in hardtanh_backward
# See README.md for why, and OPENCL-PERF.md for the
# measurements behind the two performance patches.
#
# Self-contained: run from anywhere. It builds a throwaway Python 3.12 venv in
# ./scratch (gitignored) with the SAME torch + pytorch_ocl the notebook image
# uses, applies the patches, and drops the rebuilt pt_ocl.so next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$HERE/scratch"
VENV="$SCRATCH/venv"

# Match the bc250-jupyterhub-opencl-k3s notebook image exactly.
TORCH_VERSION="2.4.0"
PYBIND11_VERSION="3.1.0"
PYTORCH_OCL_WHL="https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"

# Upstream commits the patches are written against (pytorch_dlprim HEAD of
# 2025-11-26 and the dlprimitives submodule it points at, 2024-09-04). Pinned
# so an upstream push cannot silently break the patches; bump deliberately,
# re-check every `git apply`, and re-run the correctness sweeps in tools/.
PYTORCH_DLPRIM_COMMIT="1af48d4966d83b5b43344288999343e690ea7037"
DLPRIMITIVES_COMMIT="ff2d590ab5f8110c1677ec11017bcd8d46618855"

mkdir -p "$SCRATCH"

if [[ -x "$VENV/bin/python3.12" ]]; then
    echo "== reusing scratch venv: $VENV  (rm -rf it to rebuild clean) =="
else
    echo "== building scratch venv: $VENV =="
    python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip
    "$VENV/bin/pip" install -q "torch==$TORCH_VERSION" \
        --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install -q "pybind11==$PYBIND11_VERSION" "$PYTORCH_OCL_WHL"
fi

echo "== fetching pytorch_dlprim @ ${PYTORCH_DLPRIM_COMMIT:0:12} (with dlprimitives submodule) =="
rm -rf "$SCRATCH/src" "$SCRATCH/build"
git init -q "$SCRATCH/src"
git -C "$SCRATCH/src" fetch -q --depth 1 \
    https://github.com/artyom-beilis/pytorch_dlprim "$PYTORCH_DLPRIM_COMMIT"
git -C "$SCRATCH/src" checkout -q FETCH_HEAD
git -C "$SCRATCH/src" submodule update -q --init --depth 1
got="$(git -C "$SCRATCH/src/dlprimitives" rev-parse HEAD)"
if [[ "$got" != "$DLPRIMITIVES_COMMIT" ]]; then
    echo "dlprimitives submodule is $got, expected $DLPRIMITIVES_COMMIT" >&2
    exit 1
fi

echo "== applying patches =="
# Correctness: rusticl has no work_group_reduce_* (a driver feature gap).
git -C "$SCRATCH/src/dlprimitives" apply "$HERE/custom_reduce.patch"
# Performance: all of the convolution work (occupancy, atomics-free planes,
# access patterns, prefetch, opt-in fp16) plus the activation dtype fix.
git -C "$SCRATCH/src/dlprimitives" apply "$HERE/winograd_ksplit.patch"
# Performance: OpenCL fma() is a software emulation on rusticl before Mesa
# 26.2; use mad(). Redundant but harmless on 26.2.
git -C "$SCRATCH/src"              apply "$HERE/gelu_mad.patch"
# Half-tensor fixes: reject non-float32 in convolution (came back as NaN),
# dtype-correct hardtanh/relu6/clamp formulas, contiguous grad in hardtanh_backward.
git -C "$SCRATCH/src"              apply "$HERE/pytorch_ocl_half_fixes.patch"

echo "== configuring =="
# -ffile-prefix-map: __FILE__ (TORCH_CHECK messages, torch's inline asserts)
# embeds the source and venv paths ~200 times; mapping scratch/ to /scratch
# makes the .so identical whatever directory this checkout lives in, so a
# release's sha256 can be reproduced. CMAKE_SKIP_RPATH for the same reason:
# the RPATH would name this venv's torch/lib, which the .so never needs -
# `import torch` has already loaded the libtorch sonames it links against.
mkdir -p "$SCRATCH/build"
cd "$SCRATCH/build"
cmake \
    -DCMAKE_PREFIX_PATH="$VENV/lib64/python3.12/site-packages/torch/share/cmake/Torch;$VENV/lib64/python3.12/site-packages/pybind11/share/cmake/pybind11" \
    -DPython3_EXECUTABLE="$VENV/bin/python3.12" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-ffile-prefix-map=$SCRATCH=/scratch" \
    -DCMAKE_SKIP_RPATH=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    "$SCRATCH/src"

echo "== building =="
# -j4, not -j$(nproc): each pt_ocl .cpp pulls in the full libtorch headers and
# peaks around 1.5 GB of RAM. On the BC-250's 16 GB unified memory a full-width
# parallel build swaps itself to a crawl. Override with JOBS=N if you have more.
make -j"${JOBS:-4}"

echo "== installing pt_ocl.so next to this script =="
install -m0644 "$SCRATCH/build/pytorch_ocl/pt_ocl.so" "$HERE/pt_ocl.so"
strip --strip-unneeded "$HERE/pt_ocl.so"

echo
echo "Built and staged: $HERE/pt_ocl.so"
echo "  - to ship it: sha256sum pt_ocl.so, attach to a GitHub release, pin in the notebook image's Dockerfile"
echo "  - to try it in the scratch venv directly:"
echo "      cp '$SCRATCH/build/pytorch_ocl/pt_ocl.so' \\"
echo "         '$VENV/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so'"
echo "      rm -f ~/.dlprimitives/cache.db   # old cache keyed to the old kernel source"
