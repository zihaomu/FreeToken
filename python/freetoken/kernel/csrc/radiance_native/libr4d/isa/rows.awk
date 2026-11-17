# One row per device kernel from llvm-readelf --notes of one code object.
#
# The metadata is msgpack printed as YAML and its keys are ALPHABETICAL, so .vgpr_count comes after
# .symbol and .sgpr_count before it. Emitting a row at .symbol prints the previous kernel's VGPR
# count. The row is flushed at .wavefront_size, which is last.
#
# WAVES is register-limited: on gfx1201 a SIMD has 1536 VGPRs, a wave32 allocates them in groups of
# 24 and a wave64 in groups of 8. So 96 VGPRs is 16 waves, 97 is 12, 128 is 10 and 241 is 5 -- the
# rounding is the whole point of the column, not a rounding error in it. LDS can cut it further and
# this does not model that. The 8 for wave64 is the quantum the script used for both before there
# was a wave-size column to tell them apart; nothing measured says it is wrong.
BEGIN { OFS = "\t" }
/^ *\.group_segment_fixed_size:/   { lds = $2 }
/^ *\.private_segment_fixed_size:/ { sc  = $2 }
/^ *\.sgpr_count:/                 { s   = $2 }
/^ *\.sgpr_spill_count:/           { ss  = $2 }
/^ *\.vgpr_count:/                 { v   = $2 }
/^ *\.vgpr_spill_count:/           { vs  = $2 }
/^ *\.max_flat_workgroup_size:/    { wg  = $2 }
/^ *\.name:/                       { nm  = $2 }
/^ *\.wavefront_size:/ {
  wf = $2
  if (nm != "") {
    q = (wf == 64) ? 8 : 24
    g = int((v + q - 1) / q) * q
    w = (g > 0) ? int(1536 / g) : 16
    if (w > 16) w = 16
    print v, s, lds, sc, vs + ss, w, wg, nm
    nm = ""
  }
}
