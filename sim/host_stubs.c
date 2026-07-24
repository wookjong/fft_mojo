/* The device's symbols, as the host sees them.
 *
 * A host program that names a task compiles the task's entry point for the
 * host as well -- it is @export'ed, so it is emitted for whatever is being
 * built -- and that copy refers to the launch symbols and the identity
 * registers. None of those exist off the device.
 *
 * Defining them here is what lets the host binary link. They are fatal rather
 * than empty because reaching one means a task was called instead of
 * launched, and a silent no-op would turn that into wrong results rather than
 * a stopped program.
 */

#include <stdio.h>
#include <stdlib.h>

static void host_side(const char *what)
{
    fprintf(stderr,
            "m2ndp: %s ran on the host. It only exists on the device -- a "
            "task is launched, not called.\n",
            what);
    abort();
}

void __m2ndp_set_task_range(unsigned long base, unsigned long size)
{
    (void)base;
    (void)size;
    host_side("__m2ndp_set_task_range");
}

void __m2ndp_launch_parallel(void (*kernel)(void))
{
    (void)kernel;
    host_side("__m2ndp_launch_parallel");
}

void __m2ndp_launch_serial(void (*kernel)(void))
{
    (void)kernel;
    host_side("__m2ndp_launch_serial");
}

/* Where a kernel reads the task's parameters. On the device this never
 * survives to a call -- the backend rewrites it into a read of the scratchpad
 * base -- but a host build still has to link. */
void __m2ndp_declare_params(void *g) { (void)g; host_side("__m2ndp_declare_params"); }

/* The indexed vector atomics. Declared without their real signatures, which
 * involve vectors and differ per element type: nothing here is ever entered,
 * and the linker only needs the names. */
void __m2ndp_vamoadd_i32(void) { host_side("__m2ndp_vamoadd_i32"); }
void __m2ndp_vamoadd_i64(void) { host_side("__m2ndp_vamoadd_i64"); }
void __m2ndp_vamoadd_f32(void) { host_side("__m2ndp_vamoadd_f32"); }
void __m2ndp_vamoadd_f64(void) { host_side("__m2ndp_vamoadd_f64"); }

int __m2ndp_global_uthread_id(void) { host_side("__m2ndp_global_uthread_id"); return 0; }
int __m2ndp_local_uthread_id(void) { host_side("__m2ndp_local_uthread_id"); return 0; }
int __m2ndp_group_size(void) { host_side("__m2ndp_group_size"); return 0; }
int __m2ndp_group_id(void) { host_side("__m2ndp_group_id"); return 0; }
