#!/bin/sh
# One row per device kernel in a build tree: VGPRs, SGPRs, LDS bytes, scratch bytes, spilled
# registers, the waves a SIMD the VGPR count alone allows, and the workgroup size.
#
#   BUILD=/path/to/build res.sh
#
# SPILL IS THE LOUD COLUMN, and SCRATCH after it: spilled registers are a memory round trip the
# source does not show, and scratch without spills is a private array the compiler could not keep
# in registers. WAVES is the other one: on gfx1201 a SIMD has 1536 VGPRs in wave32 allocated in
# groups of 24, so 96 VGPRs is 16 waves, 97 is 12 and 256 is 5 -- and LDS can cut it further, which
# this column does NOT model.
#
# It fails rather than printing an empty table: no .hip.o under BUILD, a code object that will not
# extract, a note read that fails, or not one kernel in the whole tree all exit 1. An instrument
# that reports "nothing to see" and exits 0 reads like a clean tree.
: "${ROCM:=/opt/rocm}"
: "${BUILD:=build}"
: "${ISA_OUT:=/tmp/r4disa}"
here=$(cd "$(dirname "$0")" && pwd)
D="$ISA_OUT/res"; rm -rf "$D"; mkdir -p "$D"
objs=$(find "$BUILD" -name '*.hip.o' 2>/dev/null | sort)
[ -n "$objs" ] || {
  echo "res.sh: no .hip.o under $BUILD" >&2
  echo "res.sh: BUILD wants a configured build tree with device kernels (RAD_WITH_HIP=ON)" >&2
  exit 1
}
for o in $objs; do
  n=$(basename "$o" .hip.o)
  # A HIP translation unit with no kernels carries no .hip_fatbin; that is not a broken object.
  "$ROCM/llvm/bin/llvm-readelf" -S "$o" 2>/dev/null | grep -q '\.hip_fatbin' || continue
  ISA_OUT="$D" "$here/co.sh" "$o" >/dev/null || {
    echo "res.sh: no gfx1201 code object in $o" >&2; exit 1; }
  "$ROCM/llvm/bin/llvm-readelf" --notes "$D/$n.co" > "$D/$n.md" || {
    echo "res.sh: llvm-readelf failed on $D/$n.co" >&2; exit 1; }
done
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' vgpr sgpr lds scratch spill waves wg kernel
rows=$(awk -f "$here/rows.awk" "$D"/*.md | sort -k5,5nr -k4,4nr -k1,1nr) || exit 1
[ -n "$rows" ] || {
  echo "res.sh: no kernel metadata in $D (every object was host-only, or the ISA_ARCH is absent)" >&2
  exit 1
}
printf '%s\n' "$rows"
