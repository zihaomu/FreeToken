#!/bin/sh
# Extract the gfx1201 device code object out of one built HIP host object.
#
#   co.sh <path/to/x.hip.o>            -> prints the path of the .co it wrote
#
# The device code is a clang offload bundle in the host object's .hip_fatbin section. The bundler
# will not read the host object directly (it finds no bundles and says so), so dump the section
# first and unbundle that. Everything else here reads the .co.
: "${ROCM:=/opt/rocm}"
: "${ISA_ARCH:=gfx1201}"
: "${ISA_OUT:=/tmp/r4disa}"
O="$1"; [ -f "$O" ] || { echo "co.sh: no such object: $O" >&2; exit 1; }
N=$(basename "$O" .hip.o); mkdir -p "$ISA_OUT"
"$ROCM/llvm/bin/llvm-objcopy" --dump-section=.hip_fatbin="$ISA_OUT/$N.fb" "$O" /dev/null 2>/dev/null \
  || { echo "co.sh: $O has no .hip_fatbin (host-only object?)" >&2; exit 1; }
"$ROCM/llvm/bin/clang-offload-bundler" --type=o --unbundle --input="$ISA_OUT/$N.fb" \
  --targets="hipv4-amdgcn-amd-amdhsa--$ISA_ARCH" --output="$ISA_OUT/$N.co" || exit 1
rm -f "$ISA_OUT/$N.fb"; echo "$ISA_OUT/$N.co"
