# movrel count per kernel, plus how many of them sit in a real loop.
#
# A `movrel` is a divergent VGPR index the compiler had to serialize; the loop that re-runs it once
# per distinct index is a `s_cbranch_execnz` back-edge around it. Counting every exec branch after
# a movrel (as this did) also counts forward branches, which are not loops, and carried the flag
# from one kernel into the next, so a kernel with no movrel at all could report loops.
#
# A branch is a back-edge when its target label has already been defined above it: llvm-objdump
# emits the blocks in address order, so "seen already" is "lower address". Forward targets are
# skipped.
function flush(  ) {
  if (n > 0) printf "%-22s %6d %6d  %s\n", tu, n, loops, k
  n = 0; loops = 0; seen = 0
}
/^[0-9a-f]+ <.*>:$/ {
  if ($2 ~ /^<L[0-9]+>:$/) lab[$2] = 1
  else { flush(); k = $2 }
  next
}
/movrel/ { n++; seen = 1 }
/s_cbranch_exec(nz|z)/ {
  # The operand is the target label (`L667`), or a bare address followed by the label in brackets
  # when the disassembler was told to symbolize; the definition line is `0000 <L667>:`.
  key = ($3 ~ /^<L/) ? $3 : "<" $2 ">:"
  if (seen && (key in lab)) loops++
  seen = 0
}
END { flush() }
