// Library half of the registered-kernel test, built at -DNK=2 and -DNK=512 and dlopen'd, which
// matches libr4d's shape: one shared object holding every kernel instantiation, entered through a
// function pointer. Both builds launch the same kernel k<0> with the same arguments; only the
// number of other instantiations in the module differs.
#include <hip/hip_runtime.h>

template <int I>
__global__ void k(float* p) { if (threadIdx.x == 0) p[blockIdx.x] = (float)I; }

/* Keeps the instantiations alive; without a use the compiler drops every unlaunched k<I>. */
static const void* g_tab[NK];

template <int I>
struct Reg {
    static void go() { g_tab[I] = (const void*)k<I>; Reg<I - 1>::go(); }
};
template <>
struct Reg<0> {
    static void go() { g_tab[0] = (const void*)k<0>; }
};

/* One thunk per instantiation. A decode step issues ~45 different kernels, each with its
 * own shim, argument buffer and host stub, so code, stub and args are cold on each launch; a
 * loop over one kernel keeps them in L1. lr_launch_distinct measures that difference. */
typedef void (*Thunk)(float*, hipStream_t);
template <int I>
static void thunk(float* p, hipStream_t st) {
    hipLaunchKernelGGL(k<I>, dim3(1), dim3(64), 0, st, p);
}
static Thunk g_thunk[NK];
template <int I>
struct RegT {
    static void go() { g_thunk[I] = &thunk<I>; RegT<I - 1>::go(); }
};
template <>
struct RegT<0> {
    static void go() { g_thunk[0] = &thunk<0>; }
};

extern "C" int lr_launch_distinct(void* p, void* stream, int n) {
    for (int i = 0; i < n; ++i) g_thunk[i % NK]((float*)p, (hipStream_t)stream);
    return 0;
}

extern "C" int lr_registered(void) {
    Reg<NK - 1>::go();
    RegT<NK - 1>::go();
    int live = 0;
    for (int i = 0; i < NK; ++i) live += g_tab[i] != 0;
    return live;
}

extern "C" int lr_launch(void* p, void* stream) {
    hipLaunchKernelGGL(k<0>, dim3(1), dim3(64), 0, (hipStream_t)stream, (float*)p);
    return 0;
}
