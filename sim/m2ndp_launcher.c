/* The m2ndp launcher: the device half of a task launch, for the NdpController
 * timing path (not spike). device_main calls these; each writes a command to
 * the fixed MMIO block and rings the doorbell, then spins on completion so the
 * call returns only once the kernel has finished. The controller recognizes the
 * MMIO accesses -- these bodies are ordinary code it runs, not intercepted at
 * entry. Link it in place of sim/launcher.c with scripts/m2ndp.lds. The launch
 * ABI is Detour's contract; include it from the submodule (-I <detour>/src). */
#include "m2ndp_launch_abi.h"

typedef unsigned long u64;

static void launch(u64 type, void (*kernel)(void)) {
  *(volatile u64 *)M2NDP_CMD_TYPE = type;
  *(volatile u64 *)M2NDP_CMD_KERNEL = (u64)kernel;
  *(volatile u64 *)M2NDP_COMPLETION = 0;  /* clear before ringing */
  *(volatile u64 *)M2NDP_DOORBELL = 1;    /* ring: request the launch */
  while (*(volatile u64 *)M2NDP_COMPLETION == 0) { /* wait until the kernel is done */ }
}

void __m2ndp_launch_parallel(void (*kernel)(void)) { launch(M2NDP_LAUNCH_PARALLEL, kernel); }
void __m2ndp_launch_serial(void (*kernel)(void)) { launch(M2NDP_LAUNCH_SERIAL, kernel); }

/* The controller already knows the data range (device_main's own arguments), so
 * this is a no-op the compiler-emitted call resolves to. */
void __m2ndp_set_task_range(u64 base, u64 size) { (void)base; (void)size; }

/* Entry the linker wants (-e _start). The controller starts device_main
 * directly, so this is never executed; it only has to exist and be valid. */
__attribute__((naked, section(".text.init"))) void _start(void) {
  __asm__ volatile("call __m2ndp_rt_launch_task\n unimp\n");
}
