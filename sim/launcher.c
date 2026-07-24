/* The launcher: the machine's half of running a task. See launcher.h. */

#include "args.h"
#include "htif.h"
#include "launcher.h"

static void say(const char *s) { htif_print(s); }

/* The memory the launcher owns. The data pool is where the host's buffers are
 * laid out; the scratchpad is the per-core region a task's globals and kernel
 * arguments live in. Neither is the task's to size -- the task assumes it owns
 * the scratchpad and the host owns the data -- so both are fixed here. */
static unsigned char pool[M2NDP_POOL_BYTES] __attribute__((aligned(64)));
static unsigned char spad[M2NDP_MAX_CORES][M2NDP_SPAD_BYTES]
    __attribute__((aligned(64)));

/* One laid-out buffer: where it landed in the pool, how big, which way. */
typedef struct {
    unsigned char *mem;
    u64 bytes;
    int is_out;
} buffer;

static buffer bufs[M2NDP_MAX_BUFS];
static int nbufs;

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
    (void)base; /* the kernels index it; Addr/Offset say where a µthread landed */

    if (size % cur_topo.packet) {
        say("the task's range is not a whole number of packets\n");
        htif_exit(2);
    }
    u64 threads = size / cur_topo.packet;
    if (threads == 0 || threads % cur_topo.cores) {
        say("microthreads do not divide evenly over the cores\n");
        htif_exit(2);
    }
    cur_topo.per_core = threads / cur_topo.cores;
}

/* ------------------------------------------------------------ setup */

static u64 align_up(u64 n, u64 a) { return (n + a - 1) / a * a; }

/* Lay the host's buffers out in the pool, back to back, and read the inputs.
 * The order is the host's declaration order, which is the order the task's
 * params struct expects -- nothing checks that they agree, which is what makes
 * it an interface. */
static int place_buffers(const m2ndp_cmdline *c)
{
    nbufs = c->nbufs;
    if (nbufs > M2NDP_MAX_BUFS) {
        say("more buffers than this build reserves for\n");
        return -1;
    }

    u64 off = 0;
    for (int i = 0; i < nbufs; i++) {
        u64 bytes = m2ndp_atou(M2NDP_BUF_BYTES(i));
        off = align_up(off, 64);
        if (off + bytes > M2NDP_POOL_BYTES) {
            say("buffers do not fit in the pool this build reserves\n");
            return -1;
        }
        bufs[i].mem = pool + off;
        bufs[i].bytes = bytes;
        bufs[i].is_out = m2ndp_atou(M2NDP_BUF_DIR(i)) != 0;
        off += bytes;

        if (bufs[i].is_out) {
            /* Zeroed rather than left as whatever was there: a finalizer that
             * accumulates would otherwise fold into stale values. */
            for (u64 j = 0; j < bytes; j++)
                bufs[i].mem[j] = 0;
        } else {
            i64 n = htif_read_file(M2NDP_BUF_FILE(i), bufs[i].mem, bytes);
            if (n < 0) {
                say("could not read ");
                say(M2NDP_BUF_FILE(i));
                say("\n");
                return -1;
            }
        }
    }
    return 0;
}

static int write_outputs(void)
{
    for (int i = 0; i < nbufs; i++) {
        if (!bufs[i].is_out)
            continue;
        if (htif_write_file(M2NDP_BUF_FILE(i), bufs[i].mem, bufs[i].bytes) < 0) {
            say("could not write ");
            say(M2NDP_BUF_FILE(i));
            say("\n");
            return -1;
        }
    }
    return 0;
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

    if (place_buffers(&c))
        return 2;

    /* Everything a launch will need, in place before the first one can
     * happen. */
    cur_topo = c.topo;

    /* Over to the device. The host's part of a launch is the range the task
     * runs over and a block of parameters; everything after that -- which
     * kernels run, in what order -- is decided in there.
     *
     * The range is `base` bytes into the first buffer: for the workloads here
     * base is zero and the range is the whole of it, but the offset is what a
     * task mapped onto part of a larger region would use. */
    if (c.argrec > M2NDP_MAX_ARGREC) {
        say("the task's parameter block has wider fields than this build "
            "reserves for\n");
        return 2;
    }
    static unsigned char block[M2NDP_MAX_BUFS * M2NDP_MAX_ARGREC];
    for (u64 i = 0; i < sizeof block; i++)
        block[i] = 0;
    for (int i = 0; i < nbufs; i++)
        *(u64 *)(block + (u64)i * c.argrec) = (u64)bufs[i].mem;
    cur_params = block;
    cur_params_bytes = (u64)nbufs * c.argrec;
    u64 range = nbufs > 0 ? (u64)bufs[0].mem + c.base : c.base;
    __m2ndp_rt_launch_task(range, c.size, (const u64 *)block);

    return write_outputs() ? 2 : 0;
}
