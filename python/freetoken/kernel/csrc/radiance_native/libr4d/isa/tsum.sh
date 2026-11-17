#!/bin/sh
# Summarise a trace.sh kernel trace over a STEADY-STATE window: where the device time goes, what
# each kernel was actually launched as, and how much of the step is not kernel time at all.
#
#   ./tsum.sh traces/dec/k_kernel_trace.csv
#   MARK=r4d_rowtopk_stage1 STEPS=20 TOP=40 ./tsum.sh traces/dec/k_kernel_trace.csv
#
# WHY A WINDOW AND NOT THE WHOLE RUN. A traced engine run is model load, then prefill, then decode,
# then whatever the harness did last, and rocprofv3's own --stats averages all of it together, so
# its per-kernel shares describe no step that ever ran. This picks a marker kernel that fires
# exactly ONCE PER DECODE STEP (the sampler's stage-1 top-k), finds the longest contiguous run of
# intervals within 25% of the median, and reports per-step numbers over STEPS of them taken from
# the middle -- steady state, with the clock ramp at the start of a run excluded.
# The marker fires once per rank per step, so the scan takes the FIRST rank it sees and ignores
# the other -- interleaving two ranks' markers would make every second interval near zero and the
# median meaningless. The window is then a wall-clock range and the table covers both ranks in it.
# If the marker never fires -- an r4d_selftest --perf run has no sampler in it -- it falls back to
# the whole trace and every number is per RUN rather than per step, which is what isolation wants.
#
# READ BUSY AND IDLE DIFFERENTLY. Kernel durations come off the hardware dispatch packet and are
# not perturbed by tracing. The gaps between them absorb the whole interception cost, so IDLE IS
# AN UPPER BOUND. OVERUS is that cost per dispatch per rank, and it is a CALIBRATION: it is the
# traced-minus-untraced step time divided by the dispatches one rank issues in a step, and it has
# to be re-derived for a configuration that differs. The ranks are separate host threads and
# inflate in parallel, so it is per-RANK dispatches that count, not the total. A traced ms/step is
# not a result -- take the step time from an untraced harness and use this to say where the time
# inside it went.
#
# BUSY IS A UNION, NOT A SUM. Two dispatches can overlap on one agent, so busy is the union of the
# intervals. The per-kernel column IS a plain sum across BOTH ranks, and pct is that kernel's share
# of all device time, so the column sums to 100% and a rank's share of it is half that.
#
# AND BUSY IS OCCUPANCY-BLIND, WHICH IS THE EASIEST WAY TO MISREAD THIS TABLE. A dispatch counts
# as busy if it is resident at all, so a 5 us kernel holding 1 of 64 CUs reads exactly like the
# GEMM saturating the card. "96% busy" therefore means the step is almost never WAITING for a
# launch -- it does NOT mean the machine is 96% utilised, and it is not an argument that there is
# no headroom. The blocks column is what to look at for that: blocks below ~64 is a kernel that
# cannot fill the device no matter how long it runs.
: "${ISA_OUT:=/tmp/r4disa}"
: "${MARK:=r4d_rowtopk_stage1}"
: "${STEPS:=20}"
: "${TOP:=32}"
: "${OVERUS:=2.0}"
F="${1:?usage: tsum.sh k_kernel_trace.csv}"
D="$ISA_OUT/trace"; rm -rf "$D"; mkdir -p "$D" || exit 1

# agent  start  end  vgpr  lds  scratch  blocks  name
# Names arrive as a full C++ signature. "(anonymous namespace)::" has to come off BEFORE the
# signature is cut at its first '(' -- otherwise every kernel in that namespace truncates to
# "void " and the whole trace lands in one empty bucket. The SPACES have to go too: the
# aggregating awk below splits on whitespace and a template argument list is full of them, so
# "<false, true, 1u," would collapse every fp8 GEMM shape into a single row.
awk -F'"' -v OFS='\t' '
  function blocks(b,   x, y, z) { x = (b[10] > 0) ? b[13] / b[10] : 0; y = (b[11] > 0) ? b[14] / b[11] : 1
                                  z = (b[12] > 0) ? b[15] / b[12] : 1; return int(x * y * z) }
  NR == 1 || $2 != "KERNEL_DISPATCH" { next }
  { split($7, b, ",")
    n = $6
    gsub(/\(anonymous namespace\)::/, "", n)
    sub(/^void /, "", n)
    sub(/\(.*$/, "", n)
    gsub(/ /, "", n)
    a = $4; gsub(/[^0-9]/, "", a)
    print a, b[3], b[4], b[7], b[5], b[6], blocks(b), n }
' "$F" > "$D/rec" || exit 1
[ -s "$D/rec" ] || { echo "tsum.sh: no KERNEL_DISPATCH rows in $F" >&2; exit 1; }
sort -k1,1n -k2,2n "$D/rec" > "$D/srec"

awk -v mark="$MARK" -v want="$STEPS" '
  { if (lo0 == 0 || $2 < lo0) lo0 = $2; if ($3 > hi0) hi0 = $3 }
  $8 ~ mark { if (ma == "") ma = $1; if ($1 == ma) m[++n] = $2 }   # ONE rank: see the note above
  END {
      if (n < 4) { print "tsum.sh: marker " mark " fired " n + 0 " times -- whole trace, per-run numbers" > "/dev/stderr"; printf "%d %d 1 run\n", lo0, hi0; exit 0 }
      for (i = 1; i < n; i++) { d[i] = m[i+1] - m[i]; s[i] = d[i] }
      for (i = 1; i < n - 1; i++) for (j = i + 1; j < n; j++) if (s[j] < s[i]) { t = s[i]; s[i] = s[j]; s[j] = t }
      med = s[int(n / 2)]
      run = 0; best = 0
      for (i = 1; i < n; i++) {
          if (d[i] > med * 0.75 && d[i] < med * 1.25) { if (++run > best) { best = run; be = i } }
          else run = 0 }
      if (best < 3) { print "tsum.sh: no steady run at median " med / 1e6 " ms -- whole trace, per-run numbers" > "/dev/stderr"; printf "%d %d 1 run\n", lo0, hi0; exit 0 }
      bs = be - best + 1
      k = (best < want) ? best : want
      lo = bs + int((best - k) / 2); hi = lo + k
      printf "%d %d %d step\n", m[lo], m[hi], k }
' "$D/srec" > "$D/win" || exit 1
read T0 T1 NSTEP MODE < "$D/win"

awk -v t0="$T0" -v t1="$T1" -v nstep="$NSTEP" -v top="$TOP" -v overus="$OVERUS" -v mode="$MODE" '
  # THE ROW IS ONE (KERNEL, GRID), NOT ONE KERNEL. One instantiation serves many shapes -- the
  # narrow fp8 GEMM runs six distinct N a step -- and averaging them together produces a rate that
  # describes none of them, which is the same mistake as averaging prefill into decode. The grid is
  # the only shape the trace carries, and for the narrow GEMM it IS N: blocks * MW * 16.
  # `min us` is there because a decision rests on the minimum, never the mean: a traced run throws
  # multi-microsecond outliers that are not independent between repeats, so a mean of them is not
  # a rate.
  $2 >= t0 && $3 <= t1 {
      dur = $3 - $2; k = $8 "\t" $7
      c[k]++; tot[k] += dur; nd++; all += dur
      if (!(k in mn) || dur < mn[k]) mn[k] = dur
      vg[k] = $4; lds[k] = $5; sc[k] = $6; bl[k] = $7; nm[k] = $8
      a = $1
      if (!(a in seen)) { seen[a] = 1; na++ }
      nda[a]++
      if ($2 > hiend[a]) { busy[a] += dur; hiend[a] = $3 }
      else if ($3 > hiend[a]) { busy[a] += $3 - hiend[a]; hiend[a] = $3 }
  }
  END {
      st = (mode == "step") ? "step" : "run"
      wall = (t1 - t0) / 1e6 / nstep
      printf "%s %d %ss  %.3f ms/%s traced  %d dispatches/%s over %d ranks (%d a rank)\n\n",
             (mode == "step") ? "window" : "whole trace,", nstep, st, wall, st,
             int(nd / nstep + 0.5), st, na, int(nd / na / nstep + 0.5)
      printf "%8s %9s %8s %8s %7s  %5s %6s %5s %7s  %s\n",
             "calls/" st, "us/" st, "avg us", "min us", "pct", "vgpr", "lds", "scr", "blocks",
             "kernel"
      for (k in c)
          printf "%8.1f %9.1f %8.1f %8.1f %6.2f%%  %5d %6d %5d %7d  %s\n",
                 c[k] / nstep, tot[k] / 1000.0 / nstep, tot[k] / 1000.0 / c[k], mn[k] / 1000.0,
                 100.0 * tot[k] / all, vg[k], lds[k], sc[k], bl[k], nm[k] | "sort -k2 -rn | head -" top
      close("sort -k2 -rn | head -" top)
      mb = 0
      printf "\n"
      for (a in seen) {
          printf "rank %d  busy %6.2f ms/%s  %5.1f%% of the traced %s\n",
                 a, busy[a] / 1e6 / nstep, st, 100.0 * busy[a] / (t1 - t0), st
          if (busy[a] > mb) mb = busy[a] }
      mb /= 1e6 * nstep
      ov = (nd / na / nstep) * overus / 1000.0
      # In whole-trace mode the idle is mostly the harness between iterations, not a launch gap,
      # so subtracting the instrument from it would be a made-up number. Say nothing instead.
      if (mode != "step") {
          printf "\nbusiest rank %.2f ms busy of %.2f ms traced. The rest is whatever the harness\n", mb, wall
          printf "did between launches, NOT a launch gap -- run the engine if you want that number.\n"
      } else {
          printf "\nbusiest rank %.2f ms/step busy, %.2f idle traced; ~%.2f of that idle is the\n",
                 mb, wall - mb, ov
          printf "instrument, so the real step is ~%.2f ms with ~%.2f ms of launch gap (%.1f%% busy).\n",
                 mb + (wall - mb - ov), wall - ov - mb, 100.0 * mb / (wall - ov)
      }
  }
' "$D/srec"
