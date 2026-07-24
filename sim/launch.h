/* Launching M²NDP kernels from a C launcher.
 *
 * The launcher is the other half of the contract in docs/INTERFACE.md. A
 * kernel is launched rather than called: its arguments are written into the
 * scratchpad rather than passed in registers, its identity arrives in
 * registers that nothing in the source mentions, and it preserves nothing.
 * All three are things C cannot say on its own, so they are said here.
 */

#ifndef M2NDP_SIM_LAUNCH_H
#define M2NDP_SIM_LAUNCH_H

typedef unsigned long u64;
typedef unsigned int u32;
typedef int i32;

/* From the link script: how much scratchpad the task's globals want. The base
 * pointer goes this far above the region, and the arguments start there. */
extern char __m2ndp_spad_size[];

/* Every value the hardware hands a microthread when it is spawned. The order
 * matches the register assignment in RISCVM2ndpArgInfo.h, which is
 * provisional -- that file and this one are the only two places that know it.
 */
typedef struct {
    u64 scratchpad_base;
    u64 offset;
    u64 addr;
    u64 ndp_id;
    u64 local_uthread_id;
    u64 global_uthread_id;
    u64 group_id;
} m2ndp_ids;

/* Run one microthread.
 *
 * Deliberately not inlined. A kernel preserves nothing, so the clobber list
 * below covers every allocatable register; asking the compiler to satisfy
 * that in the middle of a caller's loop leaves it nowhere to keep the loop's
 * own values. In a function of its own the ordinary prologue handles it, and
 * the generated code is what you would have written by hand -- the saves and
 * reloads the compiler emits around the call are exactly the register file.
 */
__attribute__((noinline)) static void m2ndp_launch(const m2ndp_ids *id,
                                                   void (*kernel)(void))
{
    register u64 a0 __asm__("a0") = id->scratchpad_base;
    register u64 a1 __asm__("a1") = id->offset;
    register u64 a2 __asm__("a2") = id->addr;
    register u64 a3 __asm__("a3") = id->ndp_id;
    register u64 a4 __asm__("a4") = id->local_uthread_id;
    register u64 a5 __asm__("a5") = id->global_uthread_id;
    register u64 a6 __asm__("a6") = id->group_id;
    /* Pinned to t0 and listed as written rather than clobbered: with every
     * other register spoken for there is nowhere else for it to live. */
    register void (*k)(void) __asm__("t0") = kernel;

    __asm__ volatile("jalr %[k]"
                     : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4),
                       "+r"(a5), "+r"(a6), "+r"(k)
                     : [k] "r"(k)
                     : "ra", "a7", "t1", "t2", "t3", "t4", "t5", "t6", "s1",
                       "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
                       "s11", "memory");
}

/* Where a task's parameters go. They are one of its scratchpad globals, so the
 * compiler decides the offset and exports it here. */
extern const i64 __m2ndp_params_offset;

static inline void *m2ndp_args(u64 base)
{
    return (void *)(base + (u64)__m2ndp_params_offset);
}

static inline u64 m2ndp_base(void *region)
{
    /* The base is where the region starts: a task's globals are laid out from
     * it, so every offset is positive. */
    return (u64)region;
}

#endif
