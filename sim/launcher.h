/* The launcher: the machine's half of running a task.
 *
 *   spike ... task.elf <cores> <interleave> <packet> <base> <size> <nbufs> \
 *             [<dir> <bytes> <file>]...
 *
 * A task no longer describes itself here. The host owns the data and says, on
 * the command line, how many buffers there are, which way each goes, how big
 * it is, and which file it lives in. The launcher lays them out in memory,
 * reads the inputs, hands control to the task's runtime entry point, and
 * writes the outputs back. Everything specific to a workload -- how many
 * buffers, how they are used, what the answer should be -- is the host's.
 *
 * One binary per task still, because the compiler assigns scratchpad offsets
 * per module on the assumption that one task owns the scratchpad. But the
 * binary is now the launcher plus one compiled task and nothing else: no
 * per-benchmark C. See docs/INTERFACE.md.
 */

#ifndef M2NDP_SIM_LAUNCHER_H
#define M2NDP_SIM_LAUNCHER_H

#include "launch.h"
#include "topology.h"

/* Ceilings the launcher reserves for. A task with more than this does not fit
 * this build; the alternative is allocation, which a bare-metal launcher has
 * no allocator for. */
#define M2NDP_MAX_BUFS 8
/* Bytes of one field of a task's parameter block. The host says what the size
 * really is; this is only what the launcher reserves room for. */
#define M2NDP_MAX_ARGREC 64
#define M2NDP_MAX_CORES 64
#define M2NDP_SPAD_BYTES (64 * 1024)
#define M2NDP_POOL_BYTES (4 * 1024 * 1024)

/* Launching a task: the runtime's entry point, compiled from the workload.
 *
 * A task conforms to `NDPTask` and gains this; nothing in a workload is
 * written to make it appear. The host's part of a launch is the range the
 * task runs over and a block of parameters whose layout the task declares --
 * one signature for every workload, so the launcher needs no per-benchmark
 * glue. */
extern void __m2ndp_rt_launch_task(u64 base, u64 size, const u64 *params);

/* The machine's half of the same launch, called by the runtime before it
 * hands over to device_main: how many microthreads the range comes to.
 * Implemented in launcher.c. */
void __m2ndp_set_task_range(u64 base, u64 size);

/* How `device_main` reaches the machine. Both run a kernel to completion
 * before returning -- launches are synchronous, and with no barrier inside a
 * kernel the launch boundary is the model's only synchronization point.
 *
 *   parallel  one microthread per packet of the task's range, spread over the
 *             cores by the topology. The kernel keys off global_uthread_id().
 *   serial    one microthread per core, for the kernels whose work is
 *             per-core rather than per-packet.
 *
 * Neither takes a size: how much work there is was settled when the task was
 * launched. The backend also reads these names -- a function whose address
 * reaches one of them is a kernel.
 *
 * Neither carries the task's parameters either. A kernel takes no arguments:
 * the block the task was launched with is copied into a core's scratchpad
 * before a kernel runs there, and the kernel reads it from its own. So there
 * is no argument list here whose length has to be agreed with a workload, and
 * a kernel reads the fields it wants by name rather than by position. */
void __m2ndp_launch_parallel(void (*kernel)(void));
void __m2ndp_launch_serial(void (*kernel)(void));

#endif
