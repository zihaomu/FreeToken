// What a VALU instruction, an LDS read and an LDS write cost next to fp8 WMMA on gfx1201, what LDS
// delivers per CU, and how the WMMA rate depends on waves per SIMD. A GEMM's K loop is a mix of
// these, and the answer decides its tile: whether the fold's arithmetic hides under the matrix work
// (it does not: VALU and WMMA issue serially on a SIMD) and how many LDS bytes a WMMA may cost
// (64 B a clock per CU, loads and stores together, so 256 B per WMMA at the WMMA rate).
//
// Every row is a kernel of eight independent WMMA chains a wave plus V VALU adds, L LDS loads and S
// LDS stores per eight WMMA, at the occupancy its LDS padding allows (every block resident, so no
// tail). It runs about a second; the result is SIMD clocks per WMMA from the elapsed time and the
// card's mean sclk (hwmon freq1_input), so it is clock-independent as long as the clock is real.
//
//   clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 wmmamix.hip -o wmmamix
//   ./wmmamix [z]
//
// With `z` every operand is zero: the multipliers barely switch, the card stays under its power
// limit at its top clock, and the per-clock costs are clean. With random operands a power limit
// holds the clock down and, at some settings, stretches it below what hwmon reports; wmmapeak.hip
// is the throughput instrument for that case.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

typedef int      i2v __attribute__((ext_vector_type(2)));
typedef float    f8v __attribute__((ext_vector_type(8)));
typedef unsigned u2v __attribute__((ext_vector_type(2)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));

// V VALU adds, L LDS loads of B bytes a lane, S ds_store_b64 per eight WMMA; C independent chains
// (8 = all independent, 1 = one dependent chain).
template <int LDS, int V, int L, int B, int S, int C>
__global__ __launch_bounds__(256) void mix(const uint32_t* src, float* out, int iters) {
    __shared__ __attribute__((aligned(16))) char pad[LDS];
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    pad[threadIdx.x] = (char)threadIdx.x;
    __syncthreads();
    const uint32_t* p = src + ((blockIdx.x & 63u) * 32u + lane) * 32u;
    i2v a[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = (i2v){ (int)p[2 * i], (int)p[2 * i + 1] };
        b[i] = (i2v){ (int)p[16 + 2 * i], (int)p[17 + 2 * i] };
    }
    f8v c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = (f8v)(0.0f);
    float f[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) f[i] = (float)(threadIdx.x + i);
    const float kk = 1.0000001f;
    typedef typename std::conditional<B == 8, u2v, u4v>::type RT;
    RT r[L > 0 ? L : 1];
#pragma unroll
    for (int i = 0; i < (L > 0 ? L : 1); ++i) r[i] = RT{};
    // A fragment read's address pattern: lane row (l & 15) at a padded pitch, half k by lane half,
    // so sixteen rows land on distinct banks.
    const uint32_t la = (uint32_t)(uintptr_t)pad + (lane & 15u) * (B == 8 ? 72u : 144u) +
                        (lane >> 4) * B + wave * 2304u;
    const uint32_t sa = (uint32_t)(uintptr_t)pad + threadIdx.x * 8u;
    const u2v sv = { threadIdx.x, lane };
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[i % C] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[i], b[(i + 3) & 7],
                                                                           c[i % C]);
#pragma unroll
            for (int v = i * V / 8; v < (i + 1) * V / 8; ++v)
                asm volatile("v_add_f32 %0, %0, %1" : "+v"(f[v & 31]) : "v"(kk));
#pragma unroll
            for (int l = i * L / 8; l < (i + 1) * L / 8; ++l) {
                if constexpr (B == 8)
                    asm volatile("ds_load_b64 %0, %1 offset:%2" : "+v"(r[l]) : "v"(la), "i"((l % 8) * 16));
                else
                    asm volatile("ds_load_b128 %0, %1 offset:%2" : "+v"(r[l]) : "v"(la), "i"((l % 4) * 32));
            }
#pragma unroll
            for (int s = i * S / 8; s < (i + 1) * S / 8; ++s)
                asm volatile("ds_store_b64 %0, %1 offset:%2" ::"v"(sa), "v"(sv), "i"((s % 8) * 2048));
        }
        if (L > 0 || S > 0) asm volatile("s_wait_dscnt 0x0" ::: "memory");
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int t = 0; t < 8; ++t) s += c[i][t];
#pragma unroll
    for (int i = 0; i < 32; ++i) s += f[i];
#pragma unroll
    for (int i = 0; i < (L > 0 ? L : 1); ++i) s += (float)(r[i].x ^ r[i].y);
    s += (float)pad[(threadIdx.x * 7u) % LDS];
    if (s == -1.0f) out[0] = s;
}

static long read_long(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return -1;
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

static std::string g_hw;   // the hwmon directory of device 0

template <int LDS, int V, int L, int B, int S, int C>
static void run(const char* name, const uint32_t* d, float* o) {
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);
    int per = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, mix<LDS, V, L, B, S, C>, 256, 0);
    const int blocks = per * prop.multiProcessorCount;
    const int iters = 20000;
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    (void)hipEventRecord(e0);
    hipLaunchKernelGGL((mix<LDS, V, L, B, S, C>), dim3(blocks), dim3(256), 0, 0, d, o, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms1 = 0;
    (void)hipEventElapsedTime(&ms1, e0, e1);
    const int reps = (int)(1000.0f / ms1) + 1;
    for (int r = 0; r < reps / 3; ++r)
        hipLaunchKernelGGL((mix<LDS, V, L, B, S, C>), dim3(blocks), dim3(256), 0, 0, d, o, iters);
    (void)hipDeviceSynchronize();
    std::atomic<bool> stop{ false };
    double mhz = 0, watt = 0;
    int n = 0;
    std::thread th([&] {
        while (!stop) {
            const long f = read_long(g_hw + "/freq1_input");
            if (f > 0) { mhz += f / 1e6; watt += read_long(g_hw + "/power1_average") / 1e6; ++n; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    (void)hipEventRecord(e0);
    for (int r = 0; r < reps; ++r)
        hipLaunchKernelGGL((mix<LDS, V, L, B, S, C>), dim3(blocks), dim3(256), 0, 0, d, o, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    stop = true;
    th.join();
    float ms = 0;
    (void)hipEventElapsedTime(&ms, e0, e1);
    const double clk = n ? mhz / n : 0.0;
    const double wmma = 8.0 * iters * (double)blocks * 8.0 * reps;
    // 64 CUs x 2 SIMDs; HIP's multiProcessorCount is the WGP count, 32.
    const double simds = 4.0 * prop.multiProcessorCount;
    const double clk_per = simds * (double)ms * 1e-3 * clk * 1e6 / wmma;
    std::printf("  %-28s %4.1f waves/SIMD  %5.0f MHz %4.0f W  %6.1f TFLOP/s  %6.2f clk/WMMA/SIMD"
                " (%5.1f%% of 8)\n",
                name, (double)blocks * 8.0 / simds, clk, n ? watt / n : 0.0,
                wmma * 8192.0 / ((double)ms * 1e9), clk_per, 800.0 / clk_per);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
}

int main(int argc, char** argv) {
    const bool zero = argc > 1 && argv[1][0] == 'z';
    char bus[64] = {};
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), 0) == hipSuccess) {
        for (char* q = bus; *q; ++q) *q = (char)std::tolower(*q);
        const std::string base = std::string("/sys/bus/pci/devices/") + bus + "/hwmon";
        if (DIR* dd = opendir(base.c_str())) {
            while (dirent* e = readdir(dd))
                if (!std::strncmp(e->d_name, "hwmon", 5)) { g_hw = base + "/" + e->d_name; break; }
            closedir(dd);
        }
    }
    if (g_hw.empty()) { std::printf("no hwmon node for device 0: the clock cannot be read\n"); return 1; }
    const size_t nw = 64 * 32 * 32;
    std::vector<uint32_t> h(nw);
    uint64_t x = 88172645463325252ull;
    for (auto& w : h) {
        uint32_t v = 0;
        for (int b = 0; b < 4; ++b) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            const uint32_t r = (uint32_t)x;
            v |= (((r & 0x80u) | ((6u + (r >> 8) % 4u) << 3) | (r & 7u)) & 0xFFu) << (8 * b);
        }
        w = zero ? 0u : v;
    }
    uint32_t* d;
    float* o;
    (void)hipMalloc(&d, nw * 4);
    (void)hipMalloc(&o, 4);
    (void)hipMemcpy(d, h.data(), nw * 4, hipMemcpyHostToDevice);
    std::printf("fp8 WMMA mixes, %s operands, sensors %s\n", zero ? "zero" : "random", g_hw.c_str());
    std::printf("occupancy (blocks a WGP set by LDS)\n");
    run<65536, 0, 0, 8, 0, 8>("wmma only", d, o);
    run<40000, 0, 0, 8, 0, 8>("wmma only", d, o);
    run<32768, 0, 0, 8, 0, 8>("wmma only", d, o);
    run<20480, 0, 0, 8, 0, 8>("wmma only", d, o);
    run<8192, 0, 0, 8, 0, 8>("wmma only", d, o);
    std::printf("dependent chains\n");
    run<40000, 0, 0, 8, 0, 1>("1 chain", d, o);
    run<40000, 0, 0, 8, 0, 4>("4 chains", d, o);
    std::printf("VALU adds per 8 WMMA\n");
    run<40000, 8, 0, 8, 0, 8>("8 valu", d, o);
    run<40000, 16, 0, 8, 0, 8>("16 valu", d, o);
    run<40000, 32, 0, 8, 0, 8>("32 valu", d, o);
    run<40000, 64, 0, 8, 0, 8>("64 valu", d, o);
    std::printf("LDS loads per 8 WMMA (a bf16 or fp8 fragment read is 256 or 512 B a wave)\n");
    run<40000, 0, 4, 8, 0, 8>("4 ds_load_b64", d, o);
    run<40000, 0, 9, 8, 0, 8>("9 ds_load_b64", d, o);
    run<40000, 0, 12, 8, 0, 8>("12 ds_load_b64", d, o);
    run<40000, 0, 16, 8, 0, 8>("16 ds_load_b64", d, o);
    run<40000, 0, 32, 8, 0, 8>("32 ds_load_b64", d, o);
    run<40000, 0, 8, 16, 0, 8>("8 ds_load_b128", d, o);
    run<40000, 0, 16, 16, 0, 8>("16 ds_load_b128", d, o);
    std::printf("LDS stores per 8 WMMA\n");
    run<40000, 0, 0, 8, 2, 8>("2 ds_store_b64", d, o);
    run<40000, 0, 0, 8, 16, 8>("16 ds_store_b64", d, o);
    std::printf("mixes\n");
    run<40000, 0, 9, 8, 2, 8>("9 load + 2 store", d, o);
    run<40000, 0, 4, 8, 1, 8>("4 load + 1 store", d, o);
    run<40000, 16, 9, 8, 2, 8>("16 valu + 9 load + 2 store", d, o);
    return 0;
}
