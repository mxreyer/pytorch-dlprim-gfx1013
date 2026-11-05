#!/usr/bin/env bash
# check-mesa-collectives.sh - does rusticl (Mesa) support the OpenCL 2.0
# work-group collectives (work_group_reduce_add() and friends)?
#
#   tools/check-mesa-collectives.sh [mesa-ref ...]
#
# Needs network access to gitlab.freedesktop.org (Mesa) and
# raw.githubusercontent.com (LLVM libclc). Default refs: main and
# mesa-26.2.3 (the version the BC-250 numbers come from). Only reads; clones
# into a temporary directory (or $WORKDIR) and prints what it finds.
#
# Why: patches/dlprimitives/00-custom-reduce.patch forces dlprimitives'
# CUSTOM_REDUCE fallback because the work-group collectives are missing on
# rusticl. This repo's README, upstream/dlprimitives-custom-reduce-autodetect.md
# and the BC-250 blog post say:
#   - rusticl does not list __opencl_c_work_group_collective_functions in
#     CL_DEVICE_OPENCL_C_FEATURES (measured on hardware with
#     tools/libclc-probe.c: Mesa 26.1.8, 26.2.2, 26.2.3);
#   - as of LLVM 22 no libclc implements them; LLVM 23's libclc adds them
#     (libclc/clc/lib/generic/collective/);
#   - so switching to Mesa's libclc fork does not remove the need for patch 00.
# This script checks those claims against source. Questions it answers:
#   1. Does rusticl ever advertise __opencl_c_work_group_collective_functions,
#      and under what condition (e.g. a libclc version or capability check)?
#   2. Does Mesa's CLC front end (src/compiler/clc) declare or enable the
#      collectives when compiling kernels?
#   3. Is there anything libclc-variant-specific ("Patched Mesa libclc") that
#      gates them, i.e. could Mesa's libclc fork change the answer?
#   4. Which LLVM libclc releases contain the collectives?
# If rusticl turns out to advertise them with LLVM >= 23's libclc, update the
# three documents above to say so (patch 00 stays needed on LLVM 22).

set -uo pipefail

REFS=("$@")
[ ${#REFS[@]} -eq 0 ] && REFS=(main mesa-26.2.3)
WORKDIR=${WORKDIR:-$(mktemp -d)}
MESA_URL=https://gitlab.freedesktop.org/mesa/mesa.git
LLVM_RAW=https://raw.githubusercontent.com/llvm/llvm-project
PAT='work_group_collective|work_group_reduce|work_group_scan|work_group_broadcast|work_group_all|work_group_any'

hr() { printf '\n==== %s ====\n' "$*"; }

hr "LLVM libclc: which release branches have the collectives"
for b in release/20.x release/21.x release/22.x release/23.x main; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 \
    "$LLVM_RAW/$b/libclc/clc/lib/generic/collective/clc_work_group_reduce.cl" 2>/dev/null)
  case $code in
    200) echo "  $b: yes (clc_work_group_reduce.cl present)";;
    404) echo "  $b: no";;
    *)   echo "  $b: could not check (HTTP $code)";;
  esac
done

for ref in "${REFS[@]}"; do
  dir="$WORKDIR/mesa-$ref"
  hr "Mesa $ref"
  if [ ! -d "$dir" ]; then
    if ! git clone -q --depth 1 --filter=blob:none --sparse --branch "$ref" \
         "$MESA_URL" "$dir"; then
      echo "  clone of $ref failed (no access to gitlab.freedesktop.org, or no such ref)"
      continue
    fi
    git -C "$dir" sparse-checkout set src/gallium/frontends/rusticl src/compiler/clc \
      src/compiler/spirv src/compiler/nir meson.build meson_options.txt VERSION
  fi
  echo "  commit: $(git -C "$dir" log -1 --format='%h %cs %s')"
  echo "  VERSION: $(cat "$dir/VERSION" 2>/dev/null)"

  echo
  echo "-- 1. rusticl: OpenCL C features it advertises (__opencl_c_* strings)"
  grep -rnoh '__opencl_c_[a-z0-9_]*' "$dir/src/gallium/frontends/rusticl" | sort | uniq -c | sed 's/^/    /'
  echo
  echo "-- 1. rusticl: every mention of collectives (with context)"
  grep -rnE -C3 "$PAT" "$dir/src/gallium/frontends/rusticl" | sed 's/^/    /' \
    || echo "    (none)"

  echo
  echo "-- 2. CLC front end / SPIR-V / NIR: mentions of collectives"
  grep -rnE "$PAT" "$dir/src/compiler/clc" "$dir/src/compiler/spirv" "$dir/src/compiler/nir" \
    | sed 's/^/    /' || echo "    (none)"

  echo
  echo "-- 2. CLC front end: how OpenCL C features/extensions reach clang"
  grep -rnE -- '-cl-ext|opencl_c_features|__opencl_c_|features_str|cl_ext' \
    "$dir/src/compiler/clc" | head -40 | sed 's/^/    /' || echo "    (none)"

  echo
  echo "-- 3. libclc variant detection (\"Patched Mesa libclc\", libclc config)"
  grep -rnE -i 'patched mesa libclc|libclc_config|insert_libclc|libclc.*version|LIBCLC' \
    "$dir/src/gallium/frontends/rusticl" "$dir/src/compiler/clc" | head -40 | sed 's/^/    /' \
    || echo "    (none)"
done

hr "Done"
echo "Sources left in $WORKDIR"
echo "Reading guide: if section 1 lists no work_group_collective feature and"
echo "section 2 finds no collectives, rusticl does not support them regardless of"
echo "libclc, and the docs' claim stands (no LLVM qualifier needed for rusticl)."
