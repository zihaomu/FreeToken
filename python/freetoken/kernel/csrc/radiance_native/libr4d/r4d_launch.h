/* r4d_launch.h -- the kernel-launch spelling for the libr4d units tests/r4d_emu runs on the host:
 * the windowed paged attention, pair_argmax, the sandwich-norm joins, expert_scale and the QSA
 * block selection.
 *
 * R4D_KNAME wraps a template kernel name containing commas, as HIP_KERNEL_NAME does for
 * hipLaunchKernelGGL. A test may define R4D_KLAUNCH to drive the same sources through a host SIMT
 * emulator; a build does not, and gets the triple-chevron launch, which the engine services via
 * hipLaunchKernel on its own queues (docs/PLUGIN.md §5).
 *
 * R4D_DYN_LDS declares a kernel's dynamic LDS, sized by the launch: `extern __shared__` in a
 * build. The emulator has no such storage class and defines its own. */
#ifndef R4D_LAUNCH_H
#define R4D_LAUNCH_H
#define R4D_KNAME(...) __VA_ARGS__
#ifndef R4D_KLAUNCH
#define R4D_KLAUNCH(kernel, grid, block, stream, ...) kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
#endif
#ifndef R4D_DYN_LDS
#define R4D_DYN_LDS(type, name) extern __shared__ type name[]
#endif
#endif
