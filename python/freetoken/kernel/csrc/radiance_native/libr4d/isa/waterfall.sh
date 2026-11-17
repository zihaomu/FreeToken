#!/bin/sh
# Every kernel in libr4d that indexes a VGPR array with an index the compiler could not prove
# uniform -- and what that costs.
#
#   BUILD=/path/to/build ./waterfall.sh
#
# WHY THIS IS ITS OWN INSTRUMENT. A divergent VGPR index is not an addressing mode on gfx12. The
# compiler emits `v_movrels_b32` inside an exec-masked loop that re-runs once per distinct index in
# the wave -- a waterfall -- and nothing about the source says so. The shape that produces one is
#
#     for (e = lane; e < n_expert; e += kWave) v[n_own++] = ...;
#     ... later ... for (i = 0; i < n_own; ++i) ... v[i] ...
#
# where `n_own` is derived from `lane`, so every `v[i]` after it indexes divergently and becomes a
# waterfall costing tens of thousands of cycles over a wave. The fix is to unroll to the
# compile-time bound so the indices are constants. res.sh cannot see any of this -- such a kernel
# has no scratch, a modest VGPR count and an unremarkable instruction count. Only the disassembly
# says it.
#
# The count is STATIC, so read it as "look here", not as a cost. A `v_movrels` with a genuinely
# uniform index is cheap and appears here too; what makes it a waterfall is the `s_cbranch_execnz`
# back-edge around it, which the second column reports. And a waterfall in a cold kernel is worth
# nothing: most of what this finds is far below the noise of a step. Check the kernel against a
# trace before touching it.
#
# It fails rather than printing a short table: no .hip.o under BUILD, a code object that will not
# extract, a disassembly that fails, or not one kernel seen all exit 1. An instrument that reports
# "no waterfalls" because it looked at nothing reads like a clean tree.
: "${ROCM:=/opt/rocm}"
: "${BUILD:=build}"
: "${ISA_OUT:=/tmp/r4disa}"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ISA_OUT"
objs=$(find "$BUILD" -name '*.hip.o' 2>/dev/null | sort)
[ -n "$objs" ] || {
  echo "waterfall.sh: no .hip.o under $BUILD" >&2
  echo "waterfall.sh: BUILD wants a configured build tree with device kernels (RAD_WITH_HIP=ON)" >&2
  exit 1
}
printf "%-22s %6s %6s  %s\n" "translation unit" "movrel" "loops" "kernel"
kernels=0
for o in $objs; do
  tu=$(basename "$o" .hip.o)
  # A HIP translation unit with no kernels carries no .hip_fatbin; that is not a broken object.
  "$ROCM/llvm/bin/llvm-readelf" -S "$o" 2>/dev/null | grep -q '\.hip_fatbin' || continue
  "$here/co.sh" "$o" >/dev/null || {
    echo "waterfall.sh: no gfx1201 code object in $o" >&2; exit 1; }
  "$ROCM/llvm/bin/llvm-objdump" -d --symbolize-operands --no-show-raw-insn \
      "$ISA_OUT/$tu.co" > "$ISA_OUT/$tu.scan.s" 2>/dev/null || {
    echo "waterfall.sh: llvm-objdump failed on $ISA_OUT/$tu.co" >&2; exit 1; }
  awk -f "$here/waterfall.awk" -v tu="$tu" "$ISA_OUT/$tu.scan.s" || exit 1
  # Empty output is a real answer here -- no kernel has a movrel -- so the check is that kernels
  # were disassembled at all.
  kernels=$((kernels + $(grep -cE '^[0-9a-f]+ <[^L]' "$ISA_OUT/$tu.scan.s" || true)))
done
[ "$kernels" -gt 0 ] || {
  echo "waterfall.sh: no kernel disassembled (every object was host-only, or ISA_ARCH is absent)" >&2
  exit 1
}
