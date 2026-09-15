#!/usr/bin/env bash
# Rebuild pytorch_ocl (pytorch_dlprim) with the gfx1013 patches applied.
# See README.md for what they do and OPENCL-PERF.md for the measurements.
#
# Self-contained: run from anywhere. Builds a Python 3.12 venv in ./scratch
# (gitignored) with torch 2.4.0 + the pytorch_ocl 0.2.0 wheel, fetches the
# pinned upstream sources, applies the patches, and leaves the rebuilt
# pt_ocl.so next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$HERE/scratch"
VENV="$SCRATCH/venv"

# The torch the wheel was built against; pybind11 for the extension build.
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
# patches/<target>/NN-*.patch, in numeric order; README.md lists what each does.
for p in "$HERE"/patches/dlprimitives/[0-9][0-9]-*.patch; do
    git -C "$SCRATCH/src/dlprimitives" apply "$p"
done
for p in "$HERE"/patches/pytorch_dlprim/[0-9][0-9]-*.patch; do
    git -C "$SCRATCH/src" apply "$p"
done

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
echo "  - to try it in the scratch venv:"
echo "      install -m0644 '$HERE/pt_ocl.so' '$VENV/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so'"
echo "      rm -f ~/.dlprimitives/cache.db   # old cache keyed to the old kernel source"
echo "      RUSTICL_ENABLE=radeonsi '$VENV/bin/python' '$HERE/tools/wino-repro.py' 10"
echo "  - to ship it: sha256sum pt_ocl.so, attach to a GitHub release"
