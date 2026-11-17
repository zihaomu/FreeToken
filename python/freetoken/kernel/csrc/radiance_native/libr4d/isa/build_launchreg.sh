#!/bin/sh
# Builds the registered-kernel test: two shared objects differing ONLY in how many template
# instantiations they hold, and a driver that dlopens both and launches k<0> from each.
#
# gfx1201 is libr4d's target architecture, so the probe is built for it and no other: a launch
# cost measured on a different part would not describe this library.
set -e
CXX=/opt/rocm/lib/llvm/bin/clang++
F="-x hip -O3 -std=c++17 --offload-arch=gfx1201 -isystem /opt/rocm/include --rocm-path=/opt/rocm"
D=$(dirname "$0")
O=${O:-$HOME}
$CXX $F -DNK=2   -fPIC -shared "$D/launchreg_lib.hip" -o "$O/liblr_few.so"
$CXX $F -DNK=512 -fPIC -shared "$D/launchreg_lib.hip" -o "$O/liblr_many.so"
$CXX $F "$D/launchreg.hip" -o "$O/launchreg" -ldl -pthread -Wl,-rpath,"$O"
echo "built in $O"
