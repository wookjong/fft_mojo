/* How microthreads are spread over NDP cores.
 *
 * The compiler never sees this. A kernel is handed a scratchpad base and an
 * identity and works from those, so "which core" is entirely the launcher's
 * decision -- which is what makes it something a functional model can vary.
 *
 * The mapping is block-cyclic, one parameter wide:
 *
 *     core(u) = (u / chunk) % cores
 *
 *     chunk = 1                round-robin: consecutive microthreads land on
 *                              different cores, the finest interleave
 *     chunk = uthreads/core    block: each core takes one contiguous run
 *     between                  block-cyclic
 *
 * Real hardware interleaves to spread a task's data across cores rather than
 * leaving one core with all of it, so `chunk` is the knob that says how
 * finely.
 *
 * What varying it is good for: the answer must not depend on it. A benchmark
 * that gives one result at one core and another at four has a bug -- most
 * likely scratchpad state leaking between cores, which is exactly the
 * contract term (one instance per core) that nothing else tests.
 */

#ifndef M2NDP_SIM_TOPOLOGY_H
#define M2NDP_SIM_TOPOLOGY_H

#include "launch.h"

typedef struct {
    u64 cores;       /* NDP cores modelled */
    u64 per_core;    /* microthreads resident on each */
    u64 chunk;       /* microthreads handed out before moving to the next core */
} m2ndp_topology;

static inline u64 m2ndp_total(const m2ndp_topology *t)
{
    return t->cores * t->per_core;
}

/* Which core a microthread runs on. */
static inline u64 m2ndp_core_of(const m2ndp_topology *t, u64 u)
{
    return (u / t->chunk) % t->cores;
}

/* Its index within that core, counting only the microthreads that share it.
 * Whole rounds come first, then the position inside the current block. */
static inline u64 m2ndp_local_of(const m2ndp_topology *t, u64 u)
{
    u64 round = u / (t->chunk * t->cores);
    return round * t->chunk + (u % t->chunk);
}

/* The identity for microthread `u` of a task, with `region` the array of
 * per-core scratchpad regions and `stride` the size of one. */
static inline m2ndp_ids m2ndp_id_of(const m2ndp_topology *t, u64 u,
                                    void *regions, u64 stride)
{
    u64 core = m2ndp_core_of(t, u);
    m2ndp_ids id = {
        .scratchpad_base = m2ndp_base((char *)regions + core * stride),
        .ndp_id = core,
        .group_id = core,
        .local_uthread_id = m2ndp_local_of(t, u),
        .global_uthread_id = u,
        .group_size = t->per_core,
    };
    return id;
}

/* The identity for microthread `l` of core `core`, for the phases that walk
 * the scratchpad rather than the data -- an initializer or a finalizer, where
 * what matters is covering the core's bins with its own microthreads. */
static inline m2ndp_ids m2ndp_id_in_core(const m2ndp_topology *t, u64 core,
                                         u64 l, void *regions, u64 stride)
{
    m2ndp_ids id = {
        .scratchpad_base = m2ndp_base((char *)regions + core * stride),
        .ndp_id = core,
        .group_id = core,
        .local_uthread_id = l,
        .global_uthread_id = core * t->per_core + l,
        .group_size = t->per_core,
    };
    return id;
}

#endif
