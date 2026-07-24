/* Reading the command line, with no libc under us.
 *
 *   spike ... task.elf <cores> <interleave> <packet> <base> <size> \
 *             <params> <params_bytes>
 *
 * How the hardware is configured, where in the pool the task is mapped, and
 * where its parameter block sits. No buffers: the pool is shared with the
 * host, so there is nothing to name, place, read in or write out.
 */

#ifndef M2NDP_SIM_ARGS_H
#define M2NDP_SIM_ARGS_H

#include "htif.h"
#include "topology.h"

/* Decimal only, and no error reporting: the caller checks the range it wants.
 * Returns 0 for anything unparseable, which every use here rejects anyway. */
static inline u64 m2ndp_atou(const char *s)
{
    u64 v = 0;
    if (!s)
        return 0;
    while (*s >= '0' && *s <= '9')
        v = v * 10 + (u64)(*s++ - '0');
    return v;
}

typedef struct {
    m2ndp_topology topo; /* cores, interleave, packet */
    u64 base;            /* where in the pool the task's work starts */
    u64 size;            /* bytes of it; divided by packet, the microthread count */
    u64 params;          /* the parameter block, also a pool address */
    u64 params_bytes;    /* how much of it to copy into each scratchpad */
} m2ndp_cmdline;

#define M2NDP_ARGC 8

/* Fills `c` from argv[1..7]. Returns 0 on success. */
static inline int m2ndp_cmdline_parse(m2ndp_cmdline *c)
{
    if (htif_argc() < M2NDP_ARGC) {
        htif_print("usage: task.elf <cores> <interleave> <packet> <base> "
                   "<size> <params> <params_bytes>\n");
        return -1;
    }
    c->topo.cores = m2ndp_atou(htif_argv(1));
    c->topo.interleave = m2ndp_atou(htif_argv(2));
    c->topo.packet = m2ndp_atou(htif_argv(3));
    c->base = m2ndp_atou(htif_argv(4));
    c->size = m2ndp_atou(htif_argv(5));
    c->params = m2ndp_atou(htif_argv(6));
    c->params_bytes = m2ndp_atou(htif_argv(7));

    if (c->topo.cores == 0 || c->topo.interleave == 0 || c->topo.packet == 0) {
        htif_print("cores, interleave and packet must all be positive\n");
        return -1;
    }
    if (c->params == 0) {
        htif_print("the task's parameter block has no address\n");
        return -1;
    }
    return 0;
}

#endif
