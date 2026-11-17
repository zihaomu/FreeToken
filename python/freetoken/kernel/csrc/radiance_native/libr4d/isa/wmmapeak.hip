// Sustained WMMA throughput on gfx1201 at the card's own clock and power limit: the ceiling a GEMM
// is measured against. isarate.hip answers a ratio of issue slots and needs no clock; this answers
// TFLOP/s, which at a power limit depends on how hard the multipliers switch, so the operands are
// random bits rotating through eight A and eight B fragments (consecutive instructions never see
// the same inputs) and the run is long enough for the clock to settle under the limit.
//
// Each wave issues eight independent accumulator chains back to back, every block of the launch is
// resident at once (the grid is the occupancy calculator's answer times the WGP count, so there is
// no tail), and launches repeat for the requested seconds after a warm-up of the same length. The
// card's own sensors (hwmon freq1_input, power1_average, temp edge) are sampled from a host thread
// during the timed window; the hwmon node is found from the HIP device's PCI bus id.
//
//   clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 wmmapeak.hip -o wmmapeak
//   ./wmmapeak [device] [seconds] [tile]
//
// A non-zero `tile` pads every block to 36864 B of unused LDS, the fp8 prefill tile's footprint
// (three blocks a WGP), to read the ceiling at that occupancy; 0 is full occupancy.
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
#include <vector>

typedef int    i2v  __attribute__((ext_vector_type(2)));
typedef int    i8v  __attribute__((ext_vector_type(8)));
typedef float  f8v  __attribute__((ext_vector_type(8)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

enum { kFp8 = 0, kBf16 = 1, kIu8 = 2 };

// Operand dwords a lane holds per fragment: two for the 8-bit forms, four for bf16.
template <int OP> struct Frag;
template <> struct Frag<kFp8>  { typedef i2v T; static constexpr int dw = 2; };
template <> struct Frag<kIu8>  { typedef i2v T; static constexpr int dw = 2; };
template <> struct Frag<kBf16> { typedef v8bf T; static constexpr int dw = 4; };

template <int OP>
__device__ __forceinline__ f8v mma(typename Frag<OP>::T a, typename Frag<OP>::T b, f8v c) {
    if constexpr (OP == kFp8) return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    if constexpr (OP == kBf16) return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    if constexpr (OP == kIu8)
        return __builtin_bit_cast(f8v, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                           true, a, true, b, __builtin_bit_cast(i8v, c), false));
}

template <int OP>
__device__ __forceinline__ typename Frag<OP>::T load_frag(const uint32_t* p) {
    typedef typename Frag<OP>::T T;
    union { uint32_t w[Frag<OP>::dw]; T f; } u;
#pragma unroll
    for (int i = 0; i < Frag<OP>::dw; ++i) u.w[i] = p[i];
    return u.f;
}

// `src` holds 16 fragments' worth of random dwords per lane (8 A, then 8 B); `out` receives a sum
// that is never -1 in practice, so the chains stay live.
template <int OP, int LDS>
__global__ __launch_bounds__(256) void peak(const uint32_t* __restrict__ src, float* out,
                                            int iters) {
    __shared__ char pad[LDS ? LDS : 4];
    pad[threadIdx.x & 3] = (char)threadIdx.x;
    typedef typename Frag<OP>::T T;
    constexpr int dw = Frag<OP>::dw;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t* p = src + ((size_t)(blockIdx.x & 63u) * 32u + lane) * 16u * dw;
    T a[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = load_frag<OP>(p + i * dw); b[i] = load_frag<OP>(p + (8 + i) * dw); }
    f8v c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = (f8v)(0.0f);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 8; ++i) c[i] = mma<OP>(a[i], b[(i + 3) & 7], c[i]);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int t = 0; t < 8; ++t) s += c[i][t];
    if (s == -1.0f) out[0] = s + (float)pad[0];
}

// ---- the card's sensors ------------------------------------------------------------------------
static std::string hwmon_dir(int dev) {
    char bus[64] = {};
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), dev) != hipSuccess) return "";
    for (char* q = bus; *q; ++q) *q = (char)std::tolower(*q);
    const std::string base = std::string("/sys/bus/pci/devices/") + bus + "/hwmon";
    DIR* d = opendir(base.c_str());
    if (!d) return "";
    std::string found;
    while (dirent* e = readdir(d))
        if (!std::strncmp(e->d_name, "hwmon", 5)) { found = base + "/" + e->d_name; break; }
    closedir(d);
    return found;
}

static long read_long(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return -1;
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

struct Sensors {
    double mhz_sum = 0, mhz_min = 1e30, mhz_max = 0, w_sum = 0, t_max = 0;
    int n = 0;
};

static void sample(const std::string& hw, std::atomic<bool>* stop, Sensors* s) {
    while (!stop->load()) {
        const long f = read_long(hw + "/freq1_input"), p = read_long(hw + "/power1_average"),
                   t = read_long(hw + "/temp1_input");
        if (f > 0) {
            const double mhz = (double)f / 1e6;
            s->mhz_sum += mhz;
            if (mhz < s->mhz_min) s->mhz_min = mhz;
            if (mhz > s->mhz_max) s->mhz_max = mhz;
            s->w_sum += p > 0 ? (double)p / 1e6 : 0.0;
            if ((double)t / 1e3 > s->t_max) s->t_max = (double)t / 1e3;
            ++s->n;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

template <int OP, int LDS>
static void run(const char* name, int dev, double secs, const uint32_t* d_src, float* d_out,
                const std::string& hw) {
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, dev);
    int per = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, peak<OP, LDS>, 256, 0);
    const int blocks = per * prop.multiProcessorCount;
    // About 20 ms a launch at any rate: calibrate on one short launch.
    int iters = 4096;
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    hipLaunchKernelGGL((peak<OP, LDS>), dim3(blocks), dim3(256), 0, 0, d_src, d_out, iters);
    (void)hipEventRecord(e0);
    hipLaunchKernelGGL((peak<OP, LDS>), dim3(blocks), dim3(256), 0, 0, d_src, d_out, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms = 0;
    (void)hipEventElapsedTime(&ms, e0, e1);
    iters = (int)((double)iters * 20.0 / (ms > 0.01f ? ms : 0.01f));
    if (iters < 256) iters = 256;
    const int launches = (int)(secs * 1000.0 / 20.0) + 1;
    // Warm-up of the same length, so the timed window starts at the settled clock.
    for (int i = 0; i < launches; ++i)
        hipLaunchKernelGGL((peak<OP, LDS>), dim3(blocks), dim3(256), 0, 0, d_src, d_out, iters);
    (void)hipDeviceSynchronize();
    Sensors s;
    std::atomic<bool> stop{false};
    std::thread th;
    if (!hw.empty()) th = std::thread(sample, hw, &stop, &s);
    (void)hipEventRecord(e0);
    for (int i = 0; i < launches; ++i)
        hipLaunchKernelGGL((peak<OP, LDS>), dim3(blocks), dim3(256), 0, 0, d_src, d_out, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    stop = true;
    if (th.joinable()) th.join();
    (void)hipEventElapsedTime(&ms, e0, e1);
    const double flop = 8192.0 * 8.0 * 8.0 * (double)iters * (double)blocks * (double)launches;
    const double tf = flop / ((double)ms * 1e9);
    const double mhz = s.n ? s.mhz_sum / s.n : 0.0;
    // Per CU per clock (64 CUs; HIP's multiProcessorCount is the WGP count, half of it).
    const double per_clk = mhz > 0 ? tf * 1e12 / (2.0 * prop.multiProcessorCount * mhz * 1e6) : 0.0;
    std::printf("  %-6s %3d blocks x 256  %8.1f ms  %7.1f TFLOP/s   sclk %6.0f MHz (min %4.0f max "
                "%4.0f)  %5.1f W  %4.0f C  %6.0f FLOP/clk/CU\n",
                name, blocks, ms, tf, mhz, s.n ? s.mhz_min : 0.0, s.mhz_max, s.n ? s.w_sum / s.n : 0.0,
                s.t_max, per_clk);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    const double secs = argc > 2 ? std::atof(argv[2]) : 3.0;
    const int tile = argc > 3 ? std::atoi(argv[3]) : 0;
    (void)hipSetDevice(dev);
    const std::string hw = hwmon_dir(dev);
    std::printf("  WMMA sustained, device %d, %.1f s timed after %.1f s warm, sensors %s, power cap "
                "%.0f W\n",
                dev, secs, secs, hw.empty() ? "(none)" : hw.c_str(),
                hw.empty() ? 0.0 : (double)read_long(hw + "/power1_cap") / 1e6);
    // Random operand bits, 64 blocks' worth so neighbouring blocks differ. The 8-bit forms take
    // every byte but E4M3's NaN codes and keep the exponent near one; bf16 likewise near one, so
    // nothing overflows over the run.
    const size_t nw = (size_t)64 * 32 * 16 * 4;
    std::vector<uint32_t> h8(nw), h16(nw), hi8(nw);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (uint32_t)x; };
    for (size_t i = 0; i < nw; ++i) {
        uint32_t w = 0;
        for (int b = 0; b < 4; ++b) {
            const uint32_t r = rnd();
            const uint32_t e = 6u + (r >> 8) % 4u;                 // exponent field 6..9
            w |= (((r & 0x80u) | (e << 3) | (r & 7u)) & 0xFFu) << (8 * b);
        }
        h8[i] = w;
        uint32_t v = 0;
        for (int hh = 0; hh < 2; ++hh) {
            const uint32_t r = rnd();
            const uint32_t e = 125u + (r >> 16) % 4u;              // 2^-2 .. 2^1
            v |= (((r & 0x8000u) | (e << 7) | (r & 0x7Fu)) & 0xFFFFu) << (16 * hh);
        }
        h16[i] = v;
        hi8[i] = rnd();
    }
    uint32_t *d8, *d16, *di8;
    float* dout;
    (void)hipMalloc(&d8, nw * 4);
    (void)hipMalloc(&d16, nw * 4);
    (void)hipMalloc(&di8, nw * 4);
    (void)hipMalloc(&dout, 4);
    (void)hipMemcpy(d8, h8.data(), nw * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(d16, h16.data(), nw * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(di8, hi8.data(), nw * 4, hipMemcpyHostToDevice);
    if (tile == 0) {
        run<kFp8, 0>("fp8", dev, secs, d8, dout, hw);
        run<kIu8, 0>("iu8", dev, secs, di8, dout, hw);
        run<kBf16, 0>("bf16", dev, secs, d16, dout, hw);
    } else {
        run<kFp8, 36864>("fp8", dev, secs, d8, dout, hw);
        run<kIu8, 36864>("iu8", dev, secs, di8, dout, hw);
        run<kBf16, 36864>("bf16", dev, secs, d16, dout, hw);
    }
    return 0;
}
