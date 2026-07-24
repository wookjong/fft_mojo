/* Reading the command line, with no libc under us.
 *
 *   spike ... task.elf <cores> <interleave> <packet> <base> <size> <nbufs> \
 *             [<dir> <bytes> <file>]...
 *
 * The machine parameters and the range come first, then a count and that many
 * buffer specs. All of it comes from the host, which owns the data: the
 * launcher reserves memory and files but decides nothing about how a
 * particular task uses them.
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

/* The fixed head of the command line, before the per-buffer specs. */
typedef struct {
    m2ndp_topology topo; /* cores, interleave, packet */
    u64 base;            /* offset into the range where the task's work starts */
    u64 size;            /* bytes of the range; base+size bounds the work */
    int nbufs;
} m2ndp_cmdline;

/* argv layout: the six scalars, then nbufs triples. A buffer's three fields
 * start here. */
#define M2NDP_HEAD 7
#define M2NDP_BUF_DIR(i) htif_argv(M2NDP_HEAD + 3 * (i) + 0)
#define M2NDP_BUF_BYTES(i) htif_argv(M2NDP_HEAD + 3 * (i) + 1)
#define M2NDP_BUF_FILE(i) htif_argv(M2NDP_HEAD + 3 * (i) + 2)

/* Fills `c` from argv[1..6]. Returns 0 on success. */
static inline int m2ndp_cmdline_parse(m2ndp_cmdline *c)
{
    if (htif_argc() < M2NDP_HEAD) {
        htif_print("usage: task.elf <cores> <interleave> <packet> <base> <size> "
                   "<nbufs> [<dir> <bytes> <file>]...\n");
        return -1;
    }
    c->topo.cores = m2ndp_atou(htif_argv(1));
    c->topo.interleave = m2ndp_atou(htif_argv(2));
    c->topo.packet = m2ndp_atou(htif_argv(3));
    c->base = m2ndp_atou(htif_argv(4));
    c->size = m2ndp_atou(htif_argv(5));
    c->nbufs = (int)m2ndp_atou(htif_argv(6));

    if (c->topo.cores == 0 || c->topo.interleave == 0 || c->topo.packet == 0) {
        htif_print("cores, interleave and packet must all be positive\n");
        return -1;
    }
    if (c->nbufs < 0 || (u64)htif_argc() < M2NDP_HEAD + 3ull * (u64)c->nbufs) {
        htif_print("fewer buffer specs than nbufs says\n");
        return -1;
    }
    return 0;
}

#endif
