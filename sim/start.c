/* Entry, and the two things that have to happen before C can run.
 *
 * No .S file: everything here is expressible in C with inline assembly, and
 * keeping it in C means the launcher is one language rather than two.
 */

#include "htif.h"

int launcher_main(void);

/* Naked because there is no stack yet, so no prologue can be emitted. */
__attribute__((naked, section(".text.init"))) void _start(void)
{
    __asm__ volatile("la   sp, __stack_top\n"
                     "call m2ndp_start\n");
}

void m2ndp_start(void)
{
    /* Bare metal comes up with the floating-point and vector units off, and
     * the first instruction that touches either traps. Whatever the M²NDP
     * runtime turns out to be, it has to do this too. */
    __asm__ volatile("csrs mstatus, %0" ::"r"((1u << 13) | (1u << 9)));

    htif_exit(launcher_main());
}
