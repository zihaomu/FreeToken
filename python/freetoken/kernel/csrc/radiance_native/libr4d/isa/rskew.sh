#!/bin/sh
# Per-kernel device time on each agent, and the gap between them. A rank-asymmetric all-reduce is
# a work imbalance somewhere ELSE, and this is what says where.
#
#   ./rskew.sh traces/dec/k_kernel_trace.csv
#
# READ THE ALL-REDUCE ROW FIRST AND READ IT BACKWARDS. The rank with MORE time inside the
# collective is the one that arrived early and waited; the slow rank is the other one. Every other
# row then says what the slow rank is doing that the fast one is not.
#
# ONLY ROWS OVER 20 ms TOTAL ARE PRINTED. Below that the gap is dominated by how many dispatches
# happened to land either side of the trace window and means nothing.
#
# WHAT A REAL IMBALANCE LOOKS LIKE: five rows well outside 1% and everything else inside it. If
# EVERY row is skewed by a similar percentage you are looking at two cards at different clocks or
# one driving a display, not at a work imbalance -- check that the rows reading VRAM-resident
# weights (hc_read_a, hc_up, lm_wmma) are symmetric before believing any of the rest.
#
# The CSV is trace.sh's. Commas inside the quoted kernel name break -F, so this splits on the
# quote: $4 is the agent, $6 the kernel name, $7 the tail, and p[3]/p[4] its start and end.
awk -F'"' '
NR>1 && NF>=7 {
  nm=$6; split($7,p,","); d=(p[4]-p[3])/1000.0; ag=$4;
  sub(/^void /,"",nm); gsub(/\(anonymous namespace\)::/,"",nm); sub(/[<(].*/,"",nm);
  if (ag=="Agent 1") { a[nm]+=d } else { b[nm]+=d }
}
END{
  printf "%-42s %10s %10s %9s %7s\n", "kernel", "agent1 ms", "agent2 ms", "gap ms", "gap%";
  for (k in a) {
    if (a[k]+b[k] < 20000) continue;
    g=(b[k]-a[k])/1000.0; m=(a[k]>b[k]?a[k]:b[k])/1000.0;
    printf "%-42s %10.1f %10.1f %9.1f %6.1f%%\n", k, a[k]/1000.0, b[k]/1000.0, g, 100.0*g/m;
  }
}' "$1" | sort -k4 -g
