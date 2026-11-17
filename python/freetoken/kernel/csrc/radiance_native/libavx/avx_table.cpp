/* avx_table.cpp -- one ISA level's table of kernel entry points, compiled once per level.
 *
 * Members, initialisers and declarations all expand from AVX_OP_LIST (avx_isa.h), so their order
 * cannot drift (a swapped entry would run the wrong op silently). An op added to the list without
 * an implementation at some level is a link error naming the symbol and the level.
 */
#include "avx_vec.h"
#include "avx_isa.h"

/* Every kernel at this level, declared. AVX_FN pastes the level's suffix on. */
#define AVX_DECL(nm) extern "C" int AVX_FN(avx_##nm)(const RadArgs*, RadStream);
AVX_OP_LIST(AVX_DECL)
#undef AVX_DECL

#define AVX_ENTRY(nm) AVX_FN(avx_##nm),

extern "C" const AvxKernelTable AVX_FN(avx_table) = {
    AVX_OP_LIST(AVX_ENTRY)
};

#undef AVX_ENTRY
