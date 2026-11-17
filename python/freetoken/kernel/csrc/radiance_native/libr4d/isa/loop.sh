#!/bin/sh
# Isolate the working loop of $ISA_OUT/one.s and classify its body.
#
#   loop.sh [anchor-regex]        default anchor: v_wmma
#
# A "largest backward branch" search finds the EPILOGUE, not the K loop -- the epilogue is a long
# unrolled store chain and the K loop is short. So anchor on the instruction that defines the loop
# and take the smallest backward branch that contains every one of them.
: "${ISA_OUT:=/tmp/r4disa}"
A="${1:-v_wmma}"
awk -v anchor="$A" -v out="$ISA_OUT" '
  /^[0-9a-f]+ <L[0-9]+>:$/ { lbl = $2; gsub(/[<>:]/, "", lbl); line[lbl] = NR; next }
  $0 ~ anchor { anc[++na] = NR }
  /s_cbranch|s_branch/ { for (i = 1; i <= NF; i++) if ($i ~ /^L[0-9]+$/) { t = $i
      if (t in line && line[t] < NR) { nl++; lo[nl] = line[t]; hi[nl] = NR; nm[nl] = t } } }
  END {
      if (na == 0) { print "loop.sh: no " anchor " in this kernel" > "/dev/stderr"; exit 1 }
      best = -1
      for (i = 1; i <= nl; i++) { c = 0
          for (j = 1; j <= na; j++) if (anc[j] >= lo[i] && anc[j] <= hi[i]) c++
          if (c == na && (best < 0 || hi[i] - lo[i] < best)) { best = hi[i] - lo[i]; bi = i } }
      if (best < 0) { print "loop.sh: no single loop holds every " anchor > "/dev/stderr"; exit 1 }
      printf "loop %s: %d instructions, %d %s\n", nm[bi], best, na, anchor
      printf "%d %d\n", lo[bi], hi[bi] > (out "/range")
  }' "$ISA_OUT/one.s" || exit 1
read LO HI < "$ISA_OUT/range"
sed -n "${LO},${HI}p" "$ISA_OUT/one.s" > "$ISA_OUT/loop.s"
awk '{ sub(/^[ \t]+/, "", $0); split($0, a, /[ \t]/); op = a[1]
       if (op == "" || op ~ /^[;\/]/) next
       if      (op ~ /^global_load/)  c = "global_load"
       else if (op ~ /^global_store/) c = "global_store"
       else if (op ~ /^ds_load/)      c = "ds_load"
       else if (op ~ /^ds_store/)     c = "ds_store"
       else if (op ~ /^scratch_/)     c = "SCRATCH"
       else if (op ~ /^v_wmma/)       c = "wmma"
       else if (op ~ /^s_wait/)       c = "s_wait"
       else if (op ~ /^s_delay/)      c = "s_delay"
       else if (op ~ /^s_barrier/)    c = "barrier"
       else if (op ~ /^v_/)           c = "valu"
       else if (op ~ /^s_/)           c = "salu"
       else                           c = "other"
       cnt[c]++; tot++; det[c "|" op]++ }
     END { for (k in cnt) printf "  %-13s %5d\n", k, cnt[k] | "sort -k2 -nr"; close("sort -k2 -nr")
           printf "  %-13s %5d\n\n  top ops:\n", "TOTAL", tot
           for (k in det) { split(k, p, "|"); printf "    %-34s %4d\n", p[2], det[k] | "sort -k2 -nr" } }' \
  "$ISA_OUT/loop.s" | head -34
