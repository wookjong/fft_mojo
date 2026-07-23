/* vector_add: c = a + b, one chunk per microthread.
 *
 *   spike ... vector_add.elf <cores> <chunk> a.bin b.bin c.bin
 *
 * No scratchpad and no atomics, so the core count cannot change the answer
 * for any interesting reason. That is what makes this the right place for the
 * launcher itself to be what is under test.
 */

#include "../launcher.h"

#define W 8 /* lanes per microthread, from benchmarks/vector_add.mojo */

#define MAX_ELEMS (1 << 16)
#define MAX_CORES 64
#define SPAD_BYTES (64 * 1024)

extern void vector_add(void);

static i32 a[MAX_ELEMS], b[MAX_ELEMS], c[MAX_ELEMS];
static unsigned char spad[MAX_CORES][SPAD_BYTES] __attribute__((aligned(64)));

static m2ndp_buffer bufs[] = {
    {M2NDP_IN, a, sizeof(a), 0, 0},
    {M2NDP_IN, b, sizeof(b), 0, 0},
    {M2NDP_OUT, c, sizeof(c), 0, 0},
};

static const m2ndp_phase phases[] = {
    {vector_add, M2NDP_OVER_DATA, {0, 1, 2, -1}},
};

const m2ndp_task m2ndp_this_task = {
    .usage = "vector_add.elf <cores> <chunk> <a.bin> <b.bin> <c.bin>",
    .bufs = bufs,
    .nbufs = 3,
    .phases = phases,
    .nphases = 1,
    .sizing_buf = 0,
    .elem_bytes = sizeof(i32),
    .elems_per_uthread = W,
    .spad = spad,
    .spad_stride = SPAD_BYTES,
    .max_cores = MAX_CORES,
};
