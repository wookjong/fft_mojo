/* The launcher, described rather than written.
 *
 * Every task does the same things: read its inputs, work out how many
 * microthreads that is, spread them over the cores, run each phase, write the
 * outputs. Only the shape differs -- which files, which kernels, how much
 * data one microthread takes. So a benchmark declares that shape and the
 * common launcher does the rest.
 *
 *   spike ... task.elf <cores> <chunk> <files...>
 *
 * One binary per task, not one for all of them: the compiler assigns
 * scratchpad offsets per module on the assumption that the module owns the
 * scratchpad, so two tasks cannot share an ELF. See docs/INTERFACE.md.
 */

#ifndef M2NDP_SIM_LAUNCHER_H
#define M2NDP_SIM_LAUNCHER_H

#include "launch.h"
#include "topology.h"

#define M2NDP_MAX_BUFS 6
#define M2NDP_MAX_ARGS 6
#define M2NDP_MAX_PHASES 4

typedef enum { M2NDP_IN, M2NDP_OUT } m2ndp_dir;

/* A file the task reads or writes, and the memory behind it. Buffers appear
 * on the command line in declaration order, after the core count and chunk. */
typedef struct {
    m2ndp_dir dir;
    void *mem;
    u64 capacity;    /* bytes of `mem` */
    u64 fixed_bytes; /* an output of a size the input does not decide; 0 means
                      * it comes out the same size as the sizing buffer */
    u64 bytes;       /* filled in at run time */
} m2ndp_buffer;

/* How a phase's microthreads are counted.
 *
 *   OVER_DATA  one per chunk of the input, spread across cores by the
 *              topology. The kernel keys off global_uthread_id.
 *   OVER_CORE  one per microthread of each core, every core covered. For the
 *              phases that walk the scratchpad rather than the data -- an
 *              initializer or a finalizer, striding by group_size from
 *              local_uthread_id.
 */
typedef enum { M2NDP_OVER_DATA, M2NDP_OVER_CORE } m2ndp_shape;

typedef struct {
    void (*kernel)(void);
    m2ndp_shape shape;
    /* Buffer indices, in the order the kernel's parameters take them.
     * Terminated by -1; a phase taking no arguments starts with -1. */
    int args[M2NDP_MAX_ARGS];
} m2ndp_phase;

typedef struct {
    const char *usage;

    m2ndp_buffer *bufs;
    int nbufs;

    const m2ndp_phase *phases;
    int nphases;

    /* Which buffer's length decides how much work there is, and how much of
     * it one microthread takes. Both come from the kernel source. */
    int sizing_buf;
    u64 elem_bytes;
    u64 elems_per_uthread;

    /* The scratchpad regions are the launcher's to provide; .spad only
     * reserves a size. One region per core, `stride` bytes apart. */
    void *spad;
    u64 spad_stride;
    u64 max_cores;
} m2ndp_task;

/* Each benchmark defines this; launcher.c runs it. */
extern const m2ndp_task m2ndp_this_task;

#endif
