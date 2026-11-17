#!/bin/sh
# Run a command under rocprofv3's kernel trace and leave a CSV that tsum.sh reads.
#
#   ./trace.sh traces/dec -- ./bin/radiance --model ... --port 8113
#   ./trace.sh traces/perf -- build/bin/r4d_selftest libr4d.so --perf 17408 5120 8
#
# WHAT IT IS FOR. Device-side, per-kernel dispatch records: begin/end timestamps off the hardware
# dispatch packet, plus the VGPR/LDS/scratch/grid of the instantiation that ACTUALLY ran. res.sh
# can only report every instantiation in the object and leave the reader to guess which one the
# engine picked; the engine's own --profile-ops is host-side and per-issue-site and restores a sync
# per call, which is what makes it oversell collectives.
#
# --kernel-trace WORKS UNMASKED; --pmc DOES NOT WORK AT ALL. There are no hardware counters
# available here: any counter set, any kernel filter, hangs in queue-sync and aborts. The abort
# that names an unsupported agent
#
#     rocprofiler_iterate_agent_supported_counters failed for agent 3 (gfx1036)
#     terminate called after throwing an instance of 'std::out_of_range'
#
# is COUNTER enumeration and appears only under --pmc. It names a display GPU integrated on the
# CPU, which has no counter support; masking it does not make --pmc work, and --kernel-trace never
# needed it masked. ROCR_VISIBLE_DEVICES=0,1 below pins the agent set to the two discrete cards and
# keeps that line out of the log, nothing more. rocprofv3 is NOT on PATH; it is $ROCM/bin/rocprofv3.
#
# THE TRACE CAN GO QUIET AND STILL LOOK COMPLETE, and nothing in the file says so. A server that
# idles before its request records the STARTUP and nothing else -- contiguous dispatch ids, a
# well-formed CSV, and not one of the decode steps that followed. Drive the workload the MOMENT
# health answers, keep the run short, and check tsum.sh's "dispatches/step" against what the model
# should issue before believing any table built from it. A live decode trace of ~110 steps is about
# 80 MB; a much smaller file has not recorded what it appears to have.
#
# IT IS NOT FREE, AND THE TWO HALVES OF THE ANSWER ARE PERTURBED DIFFERENTLY. Interception costs a
# couple of microseconds per dispatch per RANK. The ranks are separate host threads and inflate in
# parallel, so it is per-rank dispatches that count, not the total. Kernel DURATIONS are hardware
# timestamps and do not move; the GAPS between them absorb all of the overhead. So read busy time
# as real and idle time as an upper bound -- tsum.sh prints both and says so. Generated text is
# unchanged under tracing, so a traced run can be pointed at a real serving configuration.
#
# TO TRACE THE ENGINE, LET THE SERVER EXIT CLEANLY. The CSV is written in rocprofv3's finalisation,
# so start the server under this script, drive it with curl, then `pkill -f "bin/radianc[e]"` --
# SIGTERM. A `pkill -9` kills the process before it finalises and there is no trace at all. Budget
# ~20 s after the kill before the file appears; a decode run writes about 85 MB a minute.
#
# ALWAYS CHECK FOR A ZOMBIE AFTERWARDS -- this script does it, and it matters because a hung
# rocprofv3 ignores its own `timeout`: the hang is in finalisation, after the alarm has fired, so
# the process can outlive its harness indefinitely.
#
# AND THE ZOMBIE'S SYMPTOM IS NOT "TRACING IS BROKEN", IT IS A CORRECTNESS FAILURE IN THE NEXT TOOL
# RUN. A traced server that survived its harness still holds both cards' VRAM, so a rad-kbench gate
# after it reports a mass of skips and FAIL lines against named kernels -- which is what a
# miscompiled kernel looks like. The tell is in the reasons: `no device memory for this case`,
# `device error` and `refused a geometry its constraints accept` on kernels nothing touched. Kill
# the zombie and the same binary passes. Check `pgrep -af bin/radiance` BEFORE believing a gate.
: "${ROCM:=/opt/rocm}"
[ $# -ge 3 ] || { echo "usage: trace.sh OUTDIR -- cmd [args...]" >&2; exit 2; }
OUT=$1; shift
[ "$1" = "--" ] || { echo "trace.sh: expected -- before the command" >&2; exit 2; }
shift
rm -rf "$OUT"; mkdir -p "$OUT" || exit 1

# A decode run writes ~85 MB of CSV a minute. Refuse rather than fill the disk out from under the
# engine.
free=$(df -Pk "$OUT" | awk 'NR==2 { print int($4 / 1024) }')
[ "${free:-0}" -ge 4096 ] || { echo "trace.sh: only ${free} MiB free under $OUT, refusing" >&2; exit 1; }

ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}" "$ROCM/bin/rocprofv3" \
    --kernel-trace --output-format csv -d "$OUT" -o k -- "$@"
rc=$?

# The hang is in finalisation; nothing else clears it.
if pgrep -f rocprofv3 >/dev/null 2>&1; then
    echo "trace.sh: rocprofv3 did not finalise, killing" >&2
    pkill -9 -f rocprofv3
fi
ls "$OUT"/k_kernel_trace.csv >/dev/null 2>&1 || { echo "trace.sh: no trace written (did the command launch any kernels?)" >&2; exit 1; }
echo "trace.sh: rc=$rc  $(wc -l < "$OUT/k_kernel_trace.csv") dispatches  $(du -h "$OUT/k_kernel_trace.csv" | cut -f1)"
exit $rc
