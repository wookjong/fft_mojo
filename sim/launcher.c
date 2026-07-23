/* The common launcher. See launcher.h for what a task declares. */

#include "args.h"
#include "htif.h"
#include "launcher.h"

static void say(const char *s) { htif_print(s); }

static int read_inputs(const m2ndp_task *t)
{
    for (int i = 0; i < t->nbufs; i++) {
        m2ndp_buffer *b = &t->bufs[i];
        if (b->dir != M2NDP_IN)
            continue;
        i64 n = htif_read_file(M2NDP_FILE(i), b->mem, b->capacity);
        if (n < 0) {
            say("could not read ");
            say(M2NDP_FILE(i));
            say(" -- missing, or larger than this build reserves\n");
            return -1;
        }
        b->bytes = (u64)n;
    }
    return 0;
}

static int write_outputs(const m2ndp_task *t)
{
    for (int i = 0; i < t->nbufs; i++) {
        m2ndp_buffer *b = &t->bufs[i];
        if (b->dir != M2NDP_OUT)
            continue;
        if (htif_write_file(M2NDP_FILE(i), b->mem, b->bytes) < 0) {
            say("could not write ");
            say(M2NDP_FILE(i));
            say("\n");
            return -1;
        }
    }
    return 0;
}

/* Outputs are zeroed rather than left as whatever the last run wrote: a
 * finalizer that accumulates would otherwise fold into stale values, and the
 * result would be right the first time and wrong afterwards. */
static int size_outputs(const m2ndp_task *t, u64 sizing_bytes)
{
    for (int i = 0; i < t->nbufs; i++) {
        m2ndp_buffer *b = &t->bufs[i];
        if (b->dir != M2NDP_OUT)
            continue;
        b->bytes = b->fixed_bytes ? b->fixed_bytes : sizing_bytes;
        if (b->bytes > b->capacity) {
            say("an output is larger than this build reserves\n");
            return -1;
        }
        for (u64 j = 0; j < b->bytes; j++)
            ((unsigned char *)b->mem)[j] = 0;
    }
    return 0;
}

/* A kernel's arguments live in its own core's region, so every core gets its
 * own copy. */
static void set_args(const m2ndp_task *t, const m2ndp_topology *topo,
                     const m2ndp_phase *p)
{
    for (u64 core = 0; core < topo->cores; core++) {
        u64 *args =
            m2ndp_args(m2ndp_base((char *)t->spad + core * t->spad_stride));
        for (int i = 0; i < M2NDP_MAX_ARGS && p->args[i] >= 0; i++)
            args[i] = (u64)t->bufs[p->args[i]].mem;
    }
}

static void run_phase(const m2ndp_task *t, const m2ndp_topology *topo,
                      const m2ndp_phase *p)
{
    set_args(t, topo, p);

    /* One microthread at a time. The contract has no barrier and combining
     * happens through atomics, so a sequential schedule is a legal one; see
     * docs/SIMULATION.md for what that means it cannot catch. */
    if (p->shape == M2NDP_OVER_DATA) {
        u64 total = m2ndp_total(topo);
        for (u64 u = 0; u < total; u++) {
            m2ndp_ids id = m2ndp_id_of(topo, u, t->spad, t->spad_stride);
            m2ndp_launch(&id, p->kernel);
        }
    } else {
        for (u64 core = 0; core < topo->cores; core++)
            for (u64 l = 0; l < topo->per_core; l++) {
                m2ndp_ids id =
                    m2ndp_id_in_core(topo, core, l, t->spad, t->spad_stride);
                m2ndp_launch(&id, p->kernel);
            }
    }
}

int launcher_main(void)
{
    const m2ndp_task *t = &m2ndp_this_task;

    m2ndp_topology topo;
    if (m2ndp_topology_from_args(&topo, t->nbufs, t->usage))
        return 2;
    if (topo.cores > t->max_cores) {
        say("more cores than this build reserves scratchpad for\n");
        return 2;
    }

    if (read_inputs(t))
        return 2;

    u64 sizing = t->bufs[t->sizing_buf].bytes;
    u64 elems = sizing / t->elem_bytes;
    if (elems == 0 || elems % t->elems_per_uthread) {
        say("input length is not a multiple of what one microthread takes\n");
        return 2;
    }

    u64 threads = elems / t->elems_per_uthread;
    if (threads % topo.cores) {
        say("microthreads do not divide evenly over the cores\n");
        return 2;
    }
    topo.per_core = threads / topo.cores;

    if (size_outputs(t, sizing))
        return 2;

    /* Phase at a time across every core, not core at a time: launches from
     * device_main are synchronous, so a phase finishes everywhere before the
     * next begins, and the scratchpad survives between them. */
    for (int i = 0; i < t->nphases; i++)
        run_phase(t, &topo, &t->phases[i]);

    return write_outputs(t) ? 2 : 0;
}
