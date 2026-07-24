/* What a microthread is told when it is spawned.
 *
 * Which core it runs on is sim/interleave.h's, mirrored from the reference.
 * This is the rest of the identity -- the values our own convention adds --
 * and the compiler never sees any of it: a kernel is handed a scratchpad base
 * and an identity and works from those.
 *
 * `local_uthread_id` and `group_size` have no counterpart in the reference.
 * They are what a kernel strides by when several microthreads share a core,
 * and they are well defined only because a launch takes an aligned range, so
 * every core's share is the same size.
 */

#ifndef M2NDP_SIM_TOPOLOGY_H
#define M2NDP_SIM_TOPOLOGY_H

#include "interleave.h"
#include "launch.h"

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

/* The identity for a serial launch: one microthread on each core, for the
 * kernels that walk the scratchpad rather than the data. It is alone on its
 * core, so its group is itself -- which is what makes a strided walk from
 * local_uthread_id() by group_size() cover the whole scratchpad. */
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
        .group_size = 1,
    };
    return id;
}

#endif
