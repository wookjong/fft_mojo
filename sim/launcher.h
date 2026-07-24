/* The launcher: the machine's half of running a task.
 *
 *   spike ... task.elf <cores> <interleave> <packet> <base> <size> \
 *             <params> <params_bytes>
 *
 * Configures the machine and hands control to the task's entry point. Moves
 * no data: the pool is shared with the host. See docs/INTERFACE.md.
 */

#ifndef M2NDP_SIM_LAUNCHER_H
#define M2NDP_SIM_LAUNCHER_H

#include "launch.h"
#include "topology.h"

/* Ceilings this build reserves for; a bare-metal launcher has no allocator.
 * The pool is not among them: it is the host's. */
#define M2NDP_MAX_CORES 64
#define M2NDP_SPAD_BYTES (64 * 1024)

/* The task's entry point, compiled from the workload. */
extern void __m2ndp_rt_launch_task(u64 base, u64 size, const u64 *params);

/* How many microthreads the range comes to. Called before device_main. */
void __m2ndp_set_task_range(u64 base, u64 size);

/* How device_main reaches the machine.
 *
 *   parallel  one microthread per packet of the range, spread over the cores
 *   serial    one microthread per core, for per-core work
 *
 * Both return once every microthread has retired, which is what makes a launch
 * synchronous -- the model's only synchronization point.
 *
 * A kernel takes no arguments: its parameters are copied into the core's
 * scratchpad first. The backend reads these names too, a function whose
 * address reaches one being a kernel. */
void __m2ndp_launch_parallel(void (*kernel)(void));
void __m2ndp_launch_serial(void (*kernel)(void));

#endif
