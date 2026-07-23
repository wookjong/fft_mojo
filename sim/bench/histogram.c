/* histogram: tally samples into core-local bins, then fold those into one.
 *
 *   spike ... histogram.elf <cores> <chunk> samples.bin hist.bin
 *
 * The benchmark the whole contract runs through: scratchpad globals at
 * compiler-assigned offsets, an indexed vector atomic, and three phases
 * sharing one scratchpad across kernel launches.
 *
 * It is also where the core count earns its keep. Each core tallies into its
 * own bins and the finalizer folds those into the output, so the answer must
 * be the same at one core and at sixteen. It will not be if the cores share a
 * scratchpad: the body would accumulate everything into one region and the
 * finalizer would fold it once per core, leaving the output a multiple of
 * what it should be. That is the contract term -- one scratchpad instance per
 * core -- and nothing else exercises it.
 */

#include "../launcher.h"

/* Both from benchmarks/histogram.mojo. */
#define BINS 256
#define UNROLL 16

#define MAX_SAMPLES (1 << 16)
#define MAX_CORES 64
#define SPAD_BYTES (64 * 1024)

extern void histogram_init(void);
extern void histogram_body(void);
extern void histogram_final(void);

static i32 samples[MAX_SAMPLES];
static i32 out[BINS];
static unsigned char spad[MAX_CORES][SPAD_BYTES] __attribute__((aligned(64)));

static m2ndp_buffer bufs[] = {
    {M2NDP_IN, samples, sizeof(samples), 0, 0},
    /* The output is one bin per value, not one per sample, so its size does
     * not follow the input's. */
    {M2NDP_OUT, out, sizeof(out), sizeof(out), 0},
};

static const m2ndp_phase phases[] = {
    /* Zero this core's bins, and fold them out again afterwards: both walk
     * the scratchpad by group_size from local_uthread_id, so both need every
     * microthread of every core. The body walks the samples instead. */
    {histogram_init, M2NDP_OVER_CORE, {-1}},
    {histogram_body, M2NDP_OVER_DATA, {0, -1}},
    {histogram_final, M2NDP_OVER_CORE, {1, -1}},
};

const m2ndp_task m2ndp_this_task = {
    .usage = "histogram.elf <cores> <chunk> <samples.bin> <hist.bin>",
    .bufs = bufs,
    .nbufs = 2,
    .phases = phases,
    .nphases = 3,
    .sizing_buf = 0,
    .elem_bytes = sizeof(i32),
    .elems_per_uthread = UNROLL,
    .spad = spad,
    .spad_stride = SPAD_BYTES,
    .max_cores = MAX_CORES,
};
