/* Which NDP core a microthread runs on.
 *
 * Mirrors M2NDP-public (fe418e8), src/m2ndp_config.h get_matched_unit_id:
 *
 *     addr / stride % units
 *
 * with the address the one that microthread was mapped to, `base + u * packet`
 * (src/ndp_unit.cc). Matching on the address rather than on the microthread's
 * index is what makes an unaligned base rotate the assignment, which a rule
 * counting from zero cannot express.
 *
 * `stride` is bytes, as the reference's is. Saying it in packets would change
 * meaning whenever the packet did.
 *
 * Separate from topology.h so it can be compiled for the host: test/ checks it
 * against a table taken from the reference. Nothing here touches the machine.
 */

#ifndef M2NDP_SIM_INTERLEAVE_H
#define M2NDP_SIM_INTERLEAVE_H

typedef unsigned long m2ndp_u64;

typedef struct {
    m2ndp_u64 cores;      /* NDP cores modelled */
    m2ndp_u64 packet;     /* bytes of the range one microthread is mapped to */
    m2ndp_u64 stride;     /* bytes handed to a core before moving to the next */
    m2ndp_u64 base;       /* where the task's range starts */
    m2ndp_u64 size;       /* bytes of it */
} m2ndp_topology;

/* Microthreads the range comes to. A trailing part-packet gets one as well,
 * as it does in the reference (ndp_unit.cc: while (count * PACKET_SIZE <
 * size)). How many land on any one core is whatever the rule below makes it
 * -- there is no promise of an even split. */
static inline m2ndp_u64 m2ndp_total(const m2ndp_topology *t)
{
    return (t->size + t->packet - 1) / t->packet;
}

/* The address microthread `u` was mapped to. */
static inline m2ndp_u64 m2ndp_addr_of(const m2ndp_topology *t, m2ndp_u64 u)
{
    return t->base + u * t->packet;
}

/* Which core it runs on. */
static inline m2ndp_u64 m2ndp_core_of(const m2ndp_topology *t, m2ndp_u64 u)
{
    return m2ndp_addr_of(t, u) / t->stride % t->cores;
}

#endif
