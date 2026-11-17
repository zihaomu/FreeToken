/* xprefetch.hip -- the expert prefetch's kernels. See xprefetch.h. */
#include <hip/hip_runtime.h>
#include <math.h>

#include "xprefetch.h"

namespace {

__device__ __forceinline__ float xp_bf(uint32_t x) { return __uint_as_float(x << 16); }

/* logits[r, e] = h[r, :] . w[e, :], bf16 in, fp32 accumulation. A block takes 16 experts, two per
 * 32-lane group. Each lane issues all its weight loads first (n_embd / 256 16-byte loads per expert,
 * held in registers) so the kernel waits on memory once; rows are staged in LDS in as few 40 KiB
 * chunks as fit (16 rows of 2560 need two). Requires n_embd % 256 == 0 and n_embd <= 4096 (the
 * launcher checks). The grid is wide on purpose: the router input is read in place and overwritten
 * once the block's GEMMs finish, so a narrower grid waiting for CUs reads it too late. */
constexpr uint32_t kLgE = 16, kLgJ = 16, kLgLds = 40u << 10;
/* On gfx9-4-generic LLVM does not unroll the row loop inside the unrolled K loop and warns per
 * copy; other targets unroll it. Silenced for this kernel only. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
__global__ __launch_bounds__(256) void xp_logits(const uint16_t* __restrict__ h,
                                                 const uint16_t* __restrict__ w, float* __restrict__ lg,
                                                 uint32_t rows, uint32_t n_embd, uint32_t ne) {
    __shared__ uint4 s_h[kLgLds / 16];
    const uint32_t lane = threadIdx.x % 32u, grp = threadIdx.x / 32u;
    const uint32_t e0 = blockIdx.x * kLgE + grp * 2u;
    const uint32_t J = n_embd / 256u;
    uint4 wv[2][kLgJ];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (uint32_t j = 0; j < kLgJ; ++j)
            if (j < J && e0 + a < ne) wv[a][j] = ((const uint4*)(w + (size_t)(e0 + a) * n_embd))[j * 32u + lane];
    float acc[2][rad::kXpMaxRows];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int r = 0; r < rad::kXpMaxRows; ++r) acc[a][r] = 0.f;
    /* A chunk is a whole number of 32-unit columns (one j) for every row. */
    const uint32_t jc = max(1u, min(J, (kLgLds / 16u) / (rows * 32u)));
    for (uint32_t j0 = 0; j0 < J; j0 += jc) {
        const uint32_t j1 = min(J, j0 + jc), w32 = (j1 - j0) * 32u;
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < rows * w32; i += blockDim.x) {
            const uint32_t r = i / w32, c = i % w32;
            s_h[r * w32 + c] = ((const uint4*)(h + (size_t)r * n_embd))[j0 * 32u + c];
        }
        __syncthreads();
#pragma unroll
        for (uint32_t j = 0; j < kLgJ; ++j) {
            if (j < j0 || j >= j1) continue;
            float wf[2][8];
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const uint32_t x[4] = { wv[a][j].x, wv[a][j].y, wv[a][j].z, wv[a][j].w };
#pragma unroll
                for (int t = 0; t < 4; ++t) { wf[a][2 * t] = xp_bf(x[t] & 0xffffu); wf[a][2 * t + 1] = xp_bf(x[t] >> 16); }
            }
#pragma unroll
            for (int r = 0; r < rad::kXpMaxRows; ++r) {
                if ((uint32_t)r >= rows) break;
                const uint4 hv = s_h[r * w32 + (j - j0) * 32u + lane];
                const uint32_t y[4] = { hv.x, hv.y, hv.z, hv.w };
                float hf[8];
#pragma unroll
                for (int t = 0; t < 4; ++t) { hf[2 * t] = xp_bf(y[t] & 0xffffu); hf[2 * t + 1] = xp_bf(y[t] >> 16); }
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    float d = 0.f;
#pragma unroll
                    for (int t = 0; t < 8; ++t) d += wf[a][t] * hf[t];
                    acc[a][r] += d;
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 2; ++a) {
        const uint32_t e = e0 + (uint32_t)a;
        /* No unroll pragma: the early exit at `rows` makes the compiler refuse it. */
        for (int r = 0; r < rad::kXpMaxRows; ++r) {
            if ((uint32_t)r >= rows) break;
            float v = acc[a][r];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
            if (lane == 0 && e < ne) lg[(size_t)r * ne + e] = v;
        }
    }
}
#pragma clang diagnostic pop

/* One block. Counts whether the step routed the group's previous records, takes each row's top-k
 * with softmax probabilities, keeps distinct experts whose entries all sit in one pinned pool slot,
 * scores each by its highest probability (x1.5 if the previous pass routed it at this layer), and
 * writes the kXpSlots best as jobs, then the mailbox number. xp_step has already restored the
 * previous records' entries. */
constexpr float kXpPrevBoost = 0.5f;
__global__ __launch_bounds__(256) void xp_plan(const float* __restrict__ lg, uint32_t rows, uint32_t ne,
                                               uint32_t k, void** const* __restrict__ map,
                                               const int32_t* __restrict__ place,
                                               const int32_t* __restrict__ cnt, rad::XpPools pools,
                                               uint64_t ring, uint64_t rs, rad::XpRec* rec, int32_t* nrec,
                                               rad::XpBox* box, uint32_t seq, rad::XpStats* stats) {
    constexpr int NC = rad::kXpMaxRows * rad::kXpMaxK;
    __shared__ int      s_e[NC];
    __shared__ float    s_pr[NC];
    __shared__ int      s_pool[NC];
    __shared__ uint64_t s_p[NC];
    __shared__ float    s_sc[NC];      /* the candidate's score, or -1 */
    __shared__ int      s_sel[rad::kXpSlots];
    __shared__ int      s_n;
    const uint32_t tid = threadIdx.x, lane = tid % warpSize, wave = tid / warpSize;
    const uint32_t nw = blockDim.x / warpSize;

    if (tid == 0) {
        const int nr = *nrec;
        for (int t = 0; t < nr; ++t) {
            const rad::XpRec& rc = rec[t];
            atomicAdd(&stats->copied, 1u);
            if (rc.cnt && rc.place >= 0 && rc.cnt[rc.place] > 0) atomicAdd(&stats->routed, 1u);
        }
    }

    /* Rank-major order: o = j * rows + r, every row's first choice before any second one. */
    for (uint32_t r = wave; r < rows; r += nw) {
        float v[16];
        float m = -INFINITY;
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t e = lane + i * warpSize;
            v[i] = e < ne ? lg[(size_t)r * ne + e] : -INFINITY;
            m = fmaxf(m, v[i]);
        }
        for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
        float sum = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) sum += v[i] > -INFINITY ? __expf(v[i] - m) : 0.f;
        for (int o = warpSize / 2; o > 0; o >>= 1) sum += __shfl_xor(sum, o);
        for (uint32_t j = 0; j < k; ++j) {
            float bv = -INFINITY;
            int bi = -1;
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (v[i] > bv) { bv = v[i]; bi = i; }
            const int be = bi >= 0 ? (int)(lane + (uint32_t)bi * warpSize) : 0x7fffffff;
            float rv = bv;
            int re = be;
            for (int o = warpSize / 2; o > 0; o >>= 1) {
                const float ov = __shfl_xor(rv, o);
                const int oe = __shfl_xor(re, o);
                if (ov > rv || (ov == rv && oe < re)) { rv = ov; re = oe; }
            }
            if (lane == 0) {
                s_e[j * rows + r] = re;
                s_pr[j * rows + r] = __expf(rv - m) / sum;
            }
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (i == bi && re == be) v[i] = -INFINITY;
        }
    }
    __syncthreads();

    const uint32_t nc = rows * k;
    for (uint32_t o = tid; o < nc; o += blockDim.x) {
        const int e = s_e[o];
        s_pool[o] = -1;
        s_p[o] = 0;
        if (e >= 0 && (uint32_t)e < ne) {
            void** ent = map[(size_t)e * rad::kXpParts];
            if (ent) {
                const uint64_t p = (uint64_t)*(void* volatile*)ent;
                s_p[o] = p;
                int pi = -1;
                for (int i = 0; i < pools.n; ++i)
                    if (p >= pools.lo[i] && p < pools.hi[i]) { pi = i; break; }
                /* All entries of the unit must sit in one pool slot: the mover patches tables one
                 * at a time, so skip a unit caught between two of its patches. */
                if (pi >= 0) {
                    const uint64_t lo = pools.lo[pi], st = pools.stride[pi];
                    const uint64_t base = lo + ((p - lo) / st) * st;
                    for (int q = 1; q < rad::kXpParts && pi >= 0; ++q) {
                        void** eq = map[(size_t)e * rad::kXpParts + q];
                        if (!eq) continue;
                        const uint64_t v = (uint64_t)*(void* volatile*)eq;
                        if (v < base || v >= base + st) pi = -1;
                    }
                }
                s_pool[o] = pi;
            }
        }
    }
    __syncthreads();
    /* The first occurrence of an expert carries its highest probability; the rest drop out. */
    for (uint32_t o = tid; o < nc; o += blockDim.x) {
        const int e = s_e[o];
        float best = s_pr[o];
        bool first = true;
        for (uint32_t u = 0; u < nc; ++u)
            if (s_e[u] == e) {
                if (u < o) first = false;
                best = fmaxf(best, s_pr[u]);
            }
        /* The counts row holds the previous pass until this layer's router runs. */
        if (first && cnt && place && e >= 0 && (uint32_t)e < ne) {
            const int pl = place[e];
            if (pl >= 0 && cnt[pl] > 0) best *= 1.f + kXpPrevBoost;
        }
        s_sc[o] = (first && s_pool[o] >= 0) ? best : -1.f;
    }
    if (tid == 0) s_n = 0;
    __syncthreads();
    for (uint32_t o = tid; o < nc; o += blockDim.x) {
        const float sc = s_sc[o];
        if (sc < 0.f) continue;
        uint32_t rank = 0;
        for (uint32_t u = 0; u < nc; ++u) {
            const float su = s_sc[u];
            if (su > sc || (su == sc && u < o)) ++rank;
        }
        if (rank < (uint32_t)rad::kXpSlots) { s_sel[rank] = (int)o; atomicAdd(&s_n, 1); }
    }
    __syncthreads();
    const int n = s_n;
    for (int t = (int)tid; t < n * rad::kXpParts; t += (int)blockDim.x) {
        const int j = t / rad::kXpParts, q = t % rad::kXpParts, o = s_sel[j], e = s_e[o], pi = s_pool[o];
        const uint64_t lo = pools.lo[pi], st = pools.stride[pi];
        const uint64_t base = lo + ((s_p[o] - lo) / st) * st;
        const uint64_t dst = ring + (uint64_t)j * rs;
        if (q == 0) {
            /* System-scope stores write through to host memory, so the tag needs only their
             * completion, not a full L2 writeback (which would flush the compute stream's dirty
             * lines every layer). */
            __hip_atomic_store(&box->jobs[j].src, base, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&box->jobs[j].dst, dst, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&box->jobs[j].bytes, st, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            rec[j].cnt = cnt;
            rec[j].place = place ? place[e] : -1;
        }
        void** ent = map[(size_t)e * rad::kXpParts + q];
        rec[j].ent[q] = ent;
        if (ent) {
            const uint64_t old = (uint64_t)*(void* volatile*)ent;
            rec[j].oldv[q] = (void*)old;
            rec[j].newv[q] = (void*)(dst + (old - base));
        }
        /* Tag after the job's stores complete, so a visible tag implies a complete job. */
        if (q == 0) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_store(&box->jobs[j].tag, (uint64_t)seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    __syncthreads();
    if (tid == 0) {
        *nrec = n;
        __hip_atomic_store(&box->n, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        atomicAdd(&stats->planned, (unsigned)n);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __hip_atomic_store((uint32_t*)&box->seq, seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

/* Compute stream, before a layer's routed GEMMs. Restores the previous layer's patches (its GEMMs
 * are earlier in the stream), then points this layer's units at the ring, only where an entry still
 * holds what the plan read (a later heat-engine move wins) and only if `done` holds the request
 * number, which the DMA list writes after its copies. */
__global__ __launch_bounds__(128) void xp_step(const rad::XpRec* __restrict__ rrec, const int32_t* __restrict__ rn,
                                               const rad::XpRec* __restrict__ prec, const int32_t* __restrict__ pn,
                                               const uint32_t* done, uint32_t seq) {
    const int t = (int)threadIdx.x;
    if (rrec && t < *rn * rad::kXpParts) {
        const rad::XpRec& r = rrec[t / rad::kXpParts];
        const int q = t % rad::kXpParts;
        if (r.ent[q])
            atomicCAS((unsigned long long*)r.ent[q], (unsigned long long)r.newv[q], (unsigned long long)r.oldv[q]);
    }
    if (!prec) return;
    if (__hip_atomic_load((uint32_t*)done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != seq) return;
    if (t >= *pn * rad::kXpParts) return;
    const rad::XpRec& r = prec[t / rad::kXpParts];
    const int q = t % rad::kXpParts;
    if (r.ent[q])
        atomicCAS((unsigned long long*)r.ent[q], (unsigned long long)r.oldv[q], (unsigned long long)r.newv[q]);
}

}  // namespace

extern "C" int rad_xp_logits_launch(const uint16_t* h, const uint16_t* w, float* logits, uint32_t rows,
                                    uint32_t n_embd, uint32_t ne, void* stream) {
    if (rows == 0 || rows > (uint32_t)rad::kXpMaxRows || n_embd % 256u || n_embd > 256u * kLgJ) return -1;
    hipLaunchKernelGGL(xp_logits, dim3((ne + kLgE - 1) / kLgE), dim3(256), 0, (hipStream_t)stream, h, w,
                       logits, rows, n_embd, ne);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int rad_xp_plan_launch(const float* logits, uint32_t rows, uint32_t ne, uint32_t k,
                                  void** const* map, const int32_t* place, const int32_t* cnt,
                                  rad::XpPools pools, uint64_t ring, uint64_t ring_stride, rad::XpRec* rec,
                                  int32_t* nrec, rad::XpBox* box, uint32_t seq, rad::XpStats* stats,
                                  void* stream) {
    if (k == 0 || k > (uint32_t)rad::kXpMaxK || ne > (uint32_t)rad::kXpMaxNe) return -1;
    hipLaunchKernelGGL(xp_plan, dim3(1), dim3(256), 0, (hipStream_t)stream, logits, rows, ne, k, map, place,
                       cnt, pools, ring, ring_stride, rec, nrec, box, seq, stats);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int rad_xp_step_launch(const rad::XpRec* rrec, const int32_t* rn, const rad::XpRec* prec,
                                  const int32_t* pn, const uint32_t* done, uint32_t seq, void* stream) {
    hipLaunchKernelGGL(xp_step, dim3(1), dim3(128), 0, (hipStream_t)stream, rrec, rn, prec, pn, done, seq);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
