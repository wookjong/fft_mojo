/* The launcher: the machine's half of running a task. See launcher.h. */

#include "args.h"
#include "htif.h"
#include "launcher.h"

static void say(const char *s) { htif_print(s); }

/* The per-core scratchpad. The data pool is not here: it is memory the host
 * and the device share, attached by the simulator, so the launcher neither
 * owns it nor moves anything through it. */
static unsigned char spad[M2NDP_MAX_CORES][M2NDP_SPAD_BYTES]
    __attribute__((aligned(64)));

/* ------------------------------------------------------------ launching
 *
 * What a launch needs that `device_main` has no way to pass: how many cores
 * there are, how the microthreads spread over them, where their scratchpads
 * live. None of that is the workload's to say. On hardware it is state the
 * machine already holds when a kernel is launched, so the model holds it the
 * same way -- set before `device_main` runs, read by the launch symbols
 * underneath it.
 */
static m2ndp_topology cur_topo;

/* The task's parameter block, and its size in bytes. Set once when the task is
 * launched and copied into a core's scratchpad before each kernel runs there:
 * a kernel takes no arguments and reads the block from its own scratchpad, so
 * every core needs its own copy. */
static const unsigned char *cur_params;
static u64 cur_params_bytes;

static void set_args(void)
{
    for (u64 core = 0; core < cur_topo.cores; core++) {
        unsigned char *dst = (unsigned char *)m2ndp_args(m2ndp_base(spad[core]));
        for (u64 i = 0; i < cur_params_bytes; i++)
            dst[i] = cur_params[i];
    }
}

/* One microthread at a time. The contract has no barrier and combining
 * happens through atomics, so a sequential schedule is a legal one; see
 * docs/SIMULATION.md for what that means it cannot catch.
 *
 * Both symbols return only once every microthread has retired, which is what
 * makes a launch synchronous. */
void __m2ndp_launch_parallel(void (*kernel)(void))
{
    set_args();

    u64 total = m2ndp_total(&cur_topo);
    for (u64 u = 0; u < total; u++) {
        m2ndp_ids id = m2ndp_id_of(&cur_topo, u, spad, M2NDP_SPAD_BYTES);
        m2ndp_launch(&id, kernel);
    }
}

/* One microthread per core, not one per resident slot: a serial kernel's work
 * is per-core -- zeroing this core's bins, folding them out again -- and doing
 * it once per slot would either repeat it or need the kernel to divide it up.
 * So group_size() is 1 here and local_uthread_id() is 0, which leaves a
 * strided walk over the scratchpad covering all of it. */
void __m2ndp_launch_serial(void (*kernel)(void))
{
    set_args();

    for (u64 core = 0; core < cur_topo.cores; core++) {
        m2ndp_ids id = m2ndp_id_serial(&cur_topo, core, spad, M2NDP_SPAD_BYTES);
        m2ndp_launch(&id, kernel);
    }
}

/* Mapping a task onto a range: the machine's half of a task launch.
 *
 * The runtime calls this before it hands over to device_main, because this is
 * what a µthread count comes from -- one per packet of the range -- and no
 * kernel can be launched until that is known. Deciding it is the machine's
 * business; deciding what to do afterwards is the runtime's, which is why the
 * two are not the same function and only this one lives here.
 *
 * A failure has nowhere to be returned to: the runtime is compiled code with
 * no error path, and the caller is a device. So it stops the run. */
void __m2ndp_set_task_range(u64 base, u64 size)
{
    /* Which core a microthread runs on is decided from the address it was
     * mapped to, so the range's own address is part of the topology. */
    cur_topo.base = base;

    if (cur_topo.stride % cur_topo.packet) {
        say("the stride is not a whole number of packets\n");
        htif_exit(2);
    }
    if (size % cur_topo.packet) {
        say("the task's range is not a whole number of packets\n");
        htif_exit(2);
    }
    /* Aligned to a whole round, so every core takes the same number of whole
     * blocks. That is what makes group_size one number and local_uthread_id a
     * dense index, and it also means no core is left without work -- which the
     * hardware model tolerates and ours, running a finalizer on every core,
     * would get wrong. */
    if (base % (cur_topo.stride * cur_topo.cores)) {
        say("the task's range does not start on a round of the interleave\n");
        htif_exit(2);
    }
    /* And a whole number of rounds of it, so the last one is not partial and
     * every core ends up with the same share. Without this the spread is
     * lopsided -- with a stride wider than the range, one core takes all of it
     * -- while per_core below would still claim an even split. */
    u64 round = cur_topo.stride * cur_topo.cores;
    if (size == 0 || size % round) {
        say("the task's range is not a whole number of interleave rounds\n");
        htif_exit(2);
    }
    cur_topo.per_core = size / cur_topo.packet / cur_topo.cores;
}

int launcher_main(void)
{
    m2ndp_cmdline c;
    if (m2ndp_cmdline_parse(&c))
        return 2;
    if (c.topo.cores > M2NDP_MAX_CORES) {
        say("more cores than this build reserves scratchpad for\n");
        return 2;
    }
    /* The task's globals -- its parameters among them -- have to fit a core's
     * region. The linker script bounds .spad too, but against a different
     * size. */
    if ((u64)__m2ndp_spad_size > sizeof(spad[0])) {
        say("the task's scratchpad does not fit this build's\n");
        return 2;
    }
    if ((u64)__m2ndp_params_offset + c.params_bytes > sizeof(spad[0])) {
        say("the task's parameter block runs past its scratchpad\n");
        return 2;
    }

    /* Everything a launch will need, in place before the first one can
     * happen. */
    cur_topo = c.topo;
    cur_params = (const unsigned char *)c.params;
    cur_params_bytes = c.params_bytes;

    /* Over to the device. Nothing is read in or written out -- the pool is
     * the same memory the host has. Only addresses cross, and the parameters
     * go by way of the scratchpad rather than through this call. */
    __m2ndp_rt_launch_task(c.base, c.size);
    return 0;
}
