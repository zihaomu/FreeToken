// gemm_iqr's forms in part 3 of r4d_gemm_iqr.hip's kIqrParts: that unit's kernels and launches for
// the forms whose place in R4D_IQR_FORMS is 3 modulo the count, compiled beside the other parts.

#define R4D_IQR_PART 3
#include "r4d_gemm_iqr.hip"
