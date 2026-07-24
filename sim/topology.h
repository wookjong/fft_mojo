/* What a microthread is told when it is spawned.
 *
 * Which core it runs on is sim/interleave.h's, mirrored from the reference.
 * This is the rest of the identity -- the values our own convention adds --
 * and the compiler never sees any of it: a kernel is handed a scratchpad base
 * and an identity and works from those.
 *
 * `local_uthread_id` has no counterpart in the reference: it is where a
 * microthread sits among the ones sharing its core, counted out as they are
 * spawned.
 */

#ifndef M2NDP_SIM_TOPOLOGY_H
#define M2NDP_SIM_TOPOLOGY_H

#include "interleave.h"
#include "launch.h"

/* The identity for microthread `u` of a task, with `regions` the array of
 * per-core scratchpad regions, `stride` the size of one, and `local` its
 * number on the core it lands on -- which the launcher counts as it spawns,
 * so it is dense however lopsided the spread turns out to be. */
static inline m2ndp_ids m2ndp_id_of(const m2ndp_topology *t, u64 u, u64 local,
                                    void *regions, u64 stride)
{
    u64 core = m2ndp_core_of(t, u);
    m2ndp_ids id = {
        .scratchpad_base = m2ndp_base((char *)regions + core * stride),
        .ndp_id = core,
        .group_id = core,
        .local_uthread_id = local,
        .global_uthread_id = u,
    };
    return id;
}

/* The identity for a serial launch: one microthread on each core, for the
 * kernels that walk the scratchpad rather than the data. It is alone on its
 * core, so it is number zero there and the whole scratchpad is its own. */
static inline m2ndp_ids m2ndp_id_serial(const m2ndp_topology *t, u64 core,
                                        void *regions, u64 stride)
{
    (void)t;
    m2ndp_ids id = {
        .scratchpad_base = m2ndp_base((char *)regions + core * stride),
        .ndp_id = core,
        .group_id = core,
        .local_uthread_id = 0,
        .global_uthread_id = core,
    };
    return id;
}

#endif
