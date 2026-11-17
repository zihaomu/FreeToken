#!/bin/sh
# Disassemble ONE kernel out of one translation unit, into $ISA_OUT/one.s.
#
#   BUILD=... dis.sh r4d_gemm_fp8a8 gemm_8a8_prefill_kernelILb0E
#
# The pattern is a substring of the MANGLED name, which is how a template instantiation is named:
# res.sh prints those, and the template parameter order is in the kernel declaration. llvm-objdump
# --symbolize-operands prints basic-block labels in the same shape as a real symbol header
# (`0000...45110 <L1201>:`), so the kernel ends at the next header that is not an <Lnnn>.
: "${ROCM:=/opt/rocm}"
: "${BUILD:=build}"
: "${ISA_OUT:=/tmp/r4disa}"
TU="$1"; PAT="$2"
here=$(cd "$(dirname "$0")" && pwd)
O=$(find "$BUILD" -name "$TU.hip.o" | head -1)
[ -n "$O" ] || { echo "dis.sh: no translation unit $TU under $BUILD" >&2; exit 1; }
if [ ! -f "$ISA_OUT/$TU.all.s" ] || [ "$O" -nt "$ISA_OUT/$TU.all.s" ]; then
  "$here/co.sh" "$O" >/dev/null || exit 1
  "$ROCM/llvm/bin/llvm-objdump" -d --symbolize-operands --no-show-raw-insn \
    "$ISA_OUT/$TU.co" > "$ISA_OUT/$TU.all.s"
fi
awk -v pat="$PAT" '
  /^[0-9a-f]+ <.*>:$/ { if ($2 !~ /^<L[0-9]+>:$/) ink = (index($0, pat) > 0) }
  ink { print }
' "$ISA_OUT/$TU.all.s" > "$ISA_OUT/one.s"
n=$(grep -c "	" "$ISA_OUT/one.s")
[ "$n" -gt 0 ] || { echo "dis.sh: no kernel matched $PAT in $TU" >&2; exit 1; }
echo "$ISA_OUT/one.s: $n instructions, $(grep -c '^[0-9a-f]* <L' "$ISA_OUT/one.s") blocks"
