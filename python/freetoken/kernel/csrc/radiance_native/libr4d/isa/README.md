# libr4d/isa: reading the machine code the kernels became

Nine scripts and six microbenchmarks. A wall-clock timer cannot tell "this kernel is slow" from
"this kernel is at the memory wall"; the disassembly plus one calibration number can, without a GPU
run. The trace tools are the exception: they need the GPU, and they are the only instruments here
that report the geometry a kernel was launched with rather than every geometry it could have had.

    BUILD=/path/to/build ./res.sh | head -40        # every kernel: vgpr/lds/scratch
    BUILD=... ./dis.sh r4d_gemm_fp8a8 gemm_8a8_prefill_kernelILb0E
    BUILD=... ./waterfall.sh                        # divergent VGPR indices, per kernel
    ./loop.sh                                       # its K loop, classified
    ./trace.sh traces/dec -- ./bin/radiance --config radiance.yaml   # then: ./tsum.sh traces/dec/k_kernel_trace.csv
    ./selftest.sh                                   # the two parsers, on fixtures (ctest: isa_tools)

`res.sh` and `waterfall.sh` exit nonzero when they looked at nothing -- no `.hip.o`, an object whose
code object will not extract or disassemble, or not one kernel in the tree. Empty output from an
instrument is a failed run, not a clean tree. A HIP translation unit that carries no kernels has no
`.hip_fatbin` and is skipped rather than failed.

`res.sh` prints mangled names; the substring `dis.sh` takes picks one template instantiation, and
the parameter order is in the kernel's declaration (`gemm_fp8a8_kernel` is
`<Full, NT, MW, FK, GATED, I8, BIAS>`, so `ILb1ELj4ELj8ELj256ELb0E` is the narrow tile at four token
tiles and eight waves; `gemm_8a8_prefill_kernelILb0E` is the fp8 prefill tile).
`loop.sh` anchors on `v_wmma` by default and takes the smallest backward branch containing every
anchor, because a largest-loop search finds the epilogue, a long unrolled store chain.

## Tools

| tool | what it answers |
|---|---|
| `co.sh` | extracts the gfx1201 code object from one built `.hip.o`; the other scripts read that |
| `dis.sh` | disassembles one template instantiation from one translation unit |
| `loop.sh` | isolates the working loop of that disassembly and classifies its body by instruction class |
| `res.sh` | one row per device kernel in a build tree: VGPRs, SGPRs, LDS, scratch, spills, waves a SIMD, workgroup size |
| `waterfall.sh` | every kernel that indexes a VGPR array with an index the compiler could not prove uniform |
| `selftest.sh` | the parsers behind `res.sh` and `waterfall.sh`, against fixtures |
| `trace.sh` | runs a command under `rocprofv3 --kernel-trace` and leaves a CSV |
| `tsum.sh` | summarises that CSV over a steady-state window: where device time goes, per (kernel, grid) |
| `gaps.sh` | where the device is idle, and which op it was waiting for |
| `rskew.sh` | per-kernel device time on each agent, and the skew between ranks |
| `isarate.hip` | what one instruction slot is worth on gfx1201 (below) |
| `graphrate.hip` | what a dispatch costs the host to submit, and what capturing a fixed sequence saves |
| `launchreg.hip` + `launchreg_lib.hip` | whether a launch costs more when the module registers more kernels, and what crossing into a `.so` costs |
| `blockcopy.hip` | contiguous versus scattered device-to-host copy, at KV-block granularity |
| `wmmapeak.hip` | sustained fp8, iu8 and bf16 WMMA TFLOP/s at the card's clock and power limit, with the sclk, power and temperature it ran at |
| `wmmamix.hip` | what a VALU instruction, an LDS read and an LDS write cost next to fp8 WMMA, and LDS bandwidth per CU |

Each script's header states its traps and how to read its output. The microbenchmarks are built by
hand: `clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 <name>.hip -o <name>` for `isarate`,
`graphrate`, `blockcopy`, `wmmapeak` and `wmmamix`, and `build_launchreg.sh` for the two-object
launchreg pair.

## Instruction slot cost on gfx1201

`isarate.hip` runs each instruction back to back in eight independent chains, so nothing stalls
but issue. It reports a ratio, which is clock-independent, so it reads no clock.

One `v_wmma_f32_16x16x16_fp8_fp8` costs about 12.9 plain VALU issue slots, and this holds with
36 KiB of LDS limiting occupancy to what a real prefill tile gets (there it measures nearer 11.6,
for a peak of about 434 TFLOP/s). A kernel's ceiling is therefore

    TFLOP/s = 434 * wmma_slots / (wmma_slots + other_instructions_issued)

with `wmma_slots = 12.9 * (wmma count)`, both counts per loop iteration from `loop.sh`.

This is a ceiling, not a prediction. A kernel cannot beat it, and a kernel far below it is held
back by something else. Issue is often not the binding constraint: cutting a kernel's non-WMMA
instruction count by more than half can make it slower, because the removed work was covering
memory and LDS latency and its removal raises register pressure, and a memory-bound kernel moves by
about a percent under the same cut. Use the model to rule a lever out, not to predict a gain.

gfx1201 has no dense 16x16x32 fp8 WMMA (`llvm-mc -mcpu=gfx1201` rejects it). The only 32-deep fp8
form is `v_swmmac_f32_16x16x32_fp8_fp8`, the 2:4 sparse one, so the GEMMs here already use the
widest dense op there is.

## Sustained WMMA rate, and what the rest of a K loop costs

`wmmapeak.hip` measures the ceiling a GEMM is held against: random operands rotating through eight
fragments, every block resident, eight seconds after eight of warm-up. At the 300 W limit, with
three blocks a WGP (the occupancy of the prefill tiles):

| op | TFLOP/s | sclk | per CU per clock |
|---|---|---|---|
| `v_wmma_f32_16x16x16_fp8_fp8` | 330 | 2536 MHz | 2037 FLOP (8 clocks a WMMA a SIMD) |
| `v_wmma_i32_16x16x16_iu8` | 334 | 2561 MHz | 2038 |
| `v_wmma_f32_16x16x16_bf16` | 170 | 2703 MHz | 984 (16 clocks) |

The power limit, not issue, sets these: with zero operands the same fp8 loop runs 418 TFLOP/s at
3.25 GHz, so how hard the multipliers switch is most of the power, and a GEMM timed on constant
operands reads fast (the fp8 prefill tile: 262 TFLOP/s on `--perfw8`'s 0x38 fill, 205 on random
codes; `--perfvideo` fills random ones). At full occupancy the fp8 run sometimes
settles at 260-270 TFLOP/s with hwmon reporting 2.8 GHz, a state in which the reported clock
overstates the delivered one; quote the three-blocks figure.

`wmmamix.hip` adds one kind of instruction at a time to eight independent WMMA chains a wave. With
zero operands (the clock is then real), against 8.36 SIMD clocks a WMMA at six waves a SIMD:

- a VALU instruction costs 1.05-1.4 SIMD clocks, serial with the WMMA (8 adds per 8 WMMA: 9.63
  clocks a WMMA; 64: 16.7). A fold's arithmetic does not hide under the matrix work; VOPD pairs
  are the only way to halve it.
- LDS serves 64 bytes a clock per CU, loads and stores alike (32 `ds_load_b64` or 16
  `ds_load_b128` per 8 WMMA: 31.5 clocks a WMMA). At the WMMA rate that is 256 bytes of LDS per
  fp8 WMMA and 512 per bf16 one. Below the limit a load costs about a clock of issue.
- nine fragment reads and two staging stores per 8 WMMA (a 1 x 8 tile staging both operands)
  run at 73% with nothing else in the loop; four and one at 90%.
- one dependent chain a wave runs as fast as eight at six waves a SIMD (96%); occupancy moves the
  rate from 94% (four waves) to 98% (twelve).

## The profiler

`rocprofv3 --pmc` does not work on this platform. With any counter set and any kernel filter, on
this tree or on `isarate` alone, it collects, then hangs in `queue.cpp:939 Timeout while waiting
for queue sync: 1 kernels still active` and aborts on SIGABRT. There are no hardware counters, so
the dynamic instruction count is not directly observable; `loop.sh` plus knowing which blocks are
predicated is how to get it.

`--kernel-trace` works without masking any device. The abort that names an unsupported agent

    rocprofiler_iterate_agent_supported_counters failed for agent 3 (gfx1036)
      :: Agent HW architecture is not supported, no counter metrics found.

comes from counter enumeration and appears only under `--pmc`. It names an integrated display GPU,
which has no counter support; masking it does not make `--pmc` work. `trace.sh` sets
`ROCR_VISIBLE_DEVICES=0,1` when it is unset, only to pin the agent set and keep that line out of
the log. It runs `$ROCM/bin/rocprofv3`, with `ROCM` defaulting to `/opt/rocm`.

**A killed `rocprofv3` can outlive its own `timeout`.** The hang is in finalisation, after the
alarm has fired, so the child ignores the timeout and waits in queue sync indefinitely. It holds no
CUs, so it does not disturb a timing, but it holds VRAM, so the next tool reports allocation
failures that look like miscompiled kernels. Follow a `--pmc` attempt with `pkill -9 -f rocprofv3`;
`trace.sh` does this itself.

### Reading a trace

* Tracing costs about 2 us per dispatch per rank. The ranks are separate host threads and slow down
  in parallel, so per-rank dispatch counts are what matter, not the total.
* Kernel durations are hardware timestamps and do not change under tracing. The gaps absorb the
  whole interception cost, so busy time is real and idle time is an upper bound. `tsum.sh` does
  that subtraction and prints both.
* Generated text does not change under tracing, so a traced run can use a real serving
  configuration rather than a proxy.
* Busy is not utilisation. A dispatch counts as busy if it is resident at all, so a 5 us kernel on
  1 of 64 CUs reads the same as a GEMM saturating the card. "96% busy" says the step is almost never
  waiting for a launch, nothing about occupancy; for that, read the blocks column, where anything
  under ~64 cannot fill the device however long it runs.
* A trace can go quiet and still look complete. A run that idles before its workload records the
  startup and nothing else, and the CSV is well formed either way. Send the workload as soon as
  health answers, keep the run short, and check `dispatches/step` in `tsum.sh`'s header against what
  the model should issue before trusting any table built from it.

## General results

Two results recur and decide which levers are worth pulling.

**Every dispatch has a fixed cost of a few microseconds.** Across the shapes a decode step asks of
the narrow fp8 GEMM, the time follows one line,

    t = fixed + bytes / achievable_stream_rate

with a fixed term of roughly 4.5 us and a stream rate near the card's achievable bandwidth. The same
fixed cost shows on every small op (rmsnorm, quant_act, add, gated_quant) whatever it moves. A
"slow" small shape is usually a fixed cost that does not amortise, not a kernel with bandwidth left,
and fewer kernels is the only lever that reaches it.

**Most weight-streaming kernels are at the memory wall.** A narrow GEMM whose K loop issues a few
hundred instructions per few kilobytes of weight per wave has about four times more issue capacity
than the card has bandwidth, and runs at the achievable stream rate. Assembly-level work on such a
kernel does not pay; confirm with `loop.sh` before starting.

**Read a kernel's own header before proposing changes to it.** Each hot kernel's source records
its ablations, sweeps and rejected variants with the reason for each, and most assembly-level ideas
are already answered there.
