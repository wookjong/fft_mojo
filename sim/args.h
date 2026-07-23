/* Reading the command line, with no libc under us.
 *
 *   spike ... task.elf <cores> <chunk> <files...>
 *
 * Topology comes in as arguments rather than as constants so that the same
 * binary can be run at one core and at four. That the two must agree is the
 * point of being able to vary it.
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

/* Fills `t` from argv[1..2]. Returns 0 on success. `files` is how many
 * arguments the task wants after those two. */
static inline int m2ndp_topology_from_args(m2ndp_topology *t, int files,
                                           const char *usage)
{
    if (htif_argc() < 3 + files) {
        htif_print("usage: ");
        htif_print(usage);
        htif_print("\n");
        return -1;
    }
    t->cores = m2ndp_atou(htif_argv(1));
    t->chunk = m2ndp_atou(htif_argv(2));
    if (t->cores == 0 || t->chunk == 0) {
        htif_print("cores and chunk must both be positive\n");
        return -1;
    }
    return 0;
}

/* The first file argument sits after the ELF, the core count and the chunk. */
#define M2NDP_FILE(i) htif_argv(3 + (i))

#endif
