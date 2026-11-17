#!/bin/sh
# WHERE THE DEVICE IS IDLE, and which op it is waiting for. tsum.sh answers "where does the device
# time go"; this answers the other half.
#
#   gaps.sh traces/dec/k_kernel_trace.csv [window_s] [agent]
#
# A gap is start(i) - max(end(0..i-1)) on ONE agent, attributed to the kernel that FOLLOWS it,
# because a kernel that starts late is the one whose dependency was not ready. Two populations come
# out and they want different fixes:
#
#   the floor      a few microseconds before nearly everything, times the dispatch count. That is
#                  the hardware launch floor plus rocprofv3's own per-dispatch interception, so
#                  most of this column is the instrument. Differences of a few tenths of a
#                  microsecond between kernels here are not results.
#   the joins      hundreds of microseconds, a handful of times a step, before the SAME few
#                  kernels. Those are host-site ops and host round trips: the issue thread stopped,
#                  so the device drained. Interception cannot manufacture a gap that size.
#
# THE CONTEXT COLUMNS ARE THE POINT. A gap before `embed_lookup` says nothing on its own; the three
# kernels BEFORE it name the join -- `rowtopk_merge | copyBuffer => embed_lookup` is a draft loop
# reading a sampled id back to the host, and `ngram_ids => gemm_bf16_nt` is the PLE's host gather
# with the host op invisible in the middle of it.
#
# RESTRICT TO A WINDOW. A traced run is model load, then prefill, then decode; over the whole span
# the busy fraction is dominated by load and means nothing. The default takes the last 3 seconds,
# which on a short generation is decode.
C="$1"; W="${2:-3}"; AG="${3:-Agent 1}"
[ -f "$C" ] || { echo "gaps.sh: no trace $C" >&2; exit 1; }
MX=$(awk -F'"' 'NR>1 && $6 != "" { split($7,p,","); if (p[4]+0>mx) mx=p[4]+0 } END { printf "%.0f", mx }' "$C")
awk -F'"' -v MX="$MX" -v W="$W" -v AG="$AG" '
  NR>1 && $6 != "" { split($7,p,","); if (p[3]+0 < MX - W*1e9) next; if ($4 != AG) next
                     n=$6; gsub(/[ \t]/,"",n); print p[3] "\t" p[4] "\t" substr(n,1,30) }' "$C" \
| sort -n \
| awk -F'\t' '
  { s=$1+0; e=$2+0; k=$3
    if (NR>1 && s > pe) { g=s-pe; tot+=g; ng++
                          if (g > 50000) { key = p2 " | " p1 " => " k; bc[key]++; bt[key]+=g/1000 }
                          else { fk[k]+=g; fn[k]++ } }
    if (e > pe) pe = e
    p2=p1; p1=k
  }
  END {
    S = "sort -k1 -nr"
    printf "idle %.1f ms over %d gaps in the window\n\nJOINS (gap > 50 us), the context is three kernels deep:\n", tot/1e6, ng
    for (q in bc) printf "  %7.1f ms %5d %8.1f us  %s\n", bt[q]/1000, bc[q], bt[q]/bc[q], q | S
    close(S)
    printf "\nFLOOR (gap <= 50 us), top by total:\n"
    for (q in fn) printf "  %7.1f ms %7d %6.2f us  %s\n", fk[q]/1e6, fn[q], fk[q]/fn[q]/1000, q | S
    close(S)
  }'
