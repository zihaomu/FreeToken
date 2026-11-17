#!/bin/sh
# The two ISA instruments' parsers, against fixtures: no build tree, no card, no ROCm.
#
#   ./selftest.sh          (ctest name: isa_tools)
#
# rows.awk is checked at the occupancy boundaries the 24-VGPR quantum makes -- 96/97, 120/121,
# 144/145 and 240/241 all change the wave count, and a formula that rounds to 8 reports the boundary
# below -- and at the wave64 quantum, which is not 24. waterfall.awk is checked on a forward branch
# after a movrel (not a loop), a real back-edge (a loop), a kernel with neither (no row) and a
# movrel with no branch after it followed by a kernel that does branch (the flag must not carry).
here=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d) || exit 1
trap 'rm -rf "$T"' EXIT INT TERM
fail=0
check() { # check <what> <expected> <actual>
  if [ "$2" = "$3" ]; then
    printf 'ok   %s\n' "$1"
  else
    printf 'FAIL %s: expected [%s], got [%s]\n' "$1" "$2" "$3" >&2
    fail=1
  fi
}

# One kernel's metadata, in llvm-readelf --notes order: alphabetical, .wavefront_size last.
metadata() { # metadata <vgpr> <wave> <name>
  printf '  .group_segment_fixed_size: 0\n'
  printf '  .private_segment_fixed_size: 0\n'
  printf '  .sgpr_count: 30\n'
  printf '  .sgpr_spill_count: 0\n'
  printf '  .vgpr_count: %s\n' "$1"
  printf '  .vgpr_spill_count: 0\n'
  printf '  .max_flat_workgroup_size: 256\n'
  printf '  .name: %s\n' "$3"
  printf '  .wavefront_size: %s\n' "$2"
}
waves() { metadata "$1" "$2" k | awk -f "$here/rows.awk" | cut -f6; }

printf '== rows.awk: wave32 VGPR quantum 24\n'
for c in 96:16 97:12 120:12 121:10 144:10 145:9 240:6 241:5 256:5; do
  check "vgpr ${c%%:*} waves/SIMD" "${c##*:}" "$(waves "${c%%:*}" 32)"
done
printf '== rows.awk: wave64 keeps quantum 8\n'
check "vgpr 97 wave64 waves/SIMD" 14 "$(waves 97 64)"
printf '== rows.awk: one row per kernel\n'
check "two kernels, two rows" 2 \
  "$( { metadata 96 32 a; metadata 97 32 b; } | awk -f "$here/rows.awk" | wc -l | tr -d ' ')"

printf '== waterfall.awk: a loop is a back-edge, not any exec branch\n'
{
  printf '0000000000001000 <_Zk_forward>:\n'
  printf '\ts_mov_b32 s0, 0\n'
  printf '\tv_movrels_b32 v1, v2\n'
  printf '\ts_cbranch_execnz L99\n'
  printf '\ts_endpgm\n'
  printf '0000000000001010 <L99>:\n'
  printf '\ts_endpgm\n'
  printf '0000000000001100 <_Zk_loop>:\n'
  printf '0000000000001104 <L5>:\n'
  printf '\tv_movrels_b32 v1, v2\n'
  printf '\ts_cbranch_execnz L5\n'
  printf '\ts_endpgm\n'
  printf '0000000000001200 <_Zk_unbranched>:\n'
  printf '\tv_movrels_b32 v1, v2\n'
  printf '\ts_endpgm\n'
  printf '0000000000001300 <_Zk_quiet>:\n'
  printf '0000000000001304 <L7>:\n'
  printf '\ts_cbranch_execnz L7\n'
  printf '\ts_endpgm\n'
} > "$T/scan.s"
out=$(awk -f "$here/waterfall.awk" -v tu=T "$T/scan.s")
col() { printf '%s\n' "$out" | awk -v k="$1" '$4 == k { print $2, $3 }'; }
check "forward branch after movrel" "1 0" "$(col '<_Zk_forward>:')"
check "back-edge after movrel" "1 1" "$(col '<_Zk_loop>:')"
check "movrel with no branch" "1 0" "$(col '<_Zk_unbranched>:')"
check "kernel with no movrel, no row" "" "$(col '<_Zk_quiet>:')"
check "flag does not carry into the next kernel" 0 \
  "$(printf '%s\n' "$out" | grep -c '_Zk_quiet>:' | tr -d ' ')"

[ "$fail" = 0 ] || exit 1
printf 'all isa parser checks passed\n'
