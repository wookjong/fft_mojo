from __future__ import annotations

"""Core FFT planning types for the M2NDP code generator -- shared by every
planning strategy in this package (fft_plan_simple.py, fft_plan_multikernel.py,
fft_plan_balanced.py, fft_plan_recursive.py).

The planner resolves every compile-time decision that affects generated code:

* supported radix selection
* stage ordering
* SIMD batch count and tail lanes
* DRAM vs. scratchpad input source
* scratchpad ping-pong bank selection
* exact input offsets
* exact stage-twiddle vectors
* inverse normalization
* exact output destination and offsets
* scratchpad allocation sizes

Host-side test input and reference values are deliberately *not* decided
here: `fft_codegen.py` emits Mojo that generates its own random input and
computes its own independent reference at runtime (see `HostPlan`), the same
way every other benchmark's `main()` does.

`fft_codegen.py` consumes the resulting plan as a read-only code-generation IR.
It must not reconstruct FFT layout, twiddle, masking, or ping-pong decisions.

This module holds: `AddressMapping`/`AddressMappingKind` (every DRAM address
formula, as data), the low-level `FFTCodegenPlan` lowering machinery
(`_build_plan`, `layouts_for_radices`, `_lower_stages`, `_make_load`/
`_make_twiddle`/`_make_store`, `_cap_max_uthread`), `_prime_factors_supported`
(the radix-factoring utility every strategy that chains kernels needs), and
`MultiKernelFFTPlan`/`MultiKernelHostPlan` plus their stride-summary tools
(produced by more than one strategy -- make_multi_kernel_plan and
make_balanced_plan both return a MultiKernelFFTPlan -- so they live here
rather than in any one strategy's own module). Every make_*_plan entry point
itself lives in its own sibling module.
"""

from dataclasses import dataclass
from enum import Enum
from math import cos, pi, sin
from typing import Literal

from codegen.fft_butterflies import SUPPORTED_RADICES


LoadSource = Literal["input", "scratchpad", "large_twiddle"]
LoadMode = Literal["vector", "scalar_pack"]
StoreDestination = Literal["output", "scratchpad"]
StoreMode = Literal["vector", "scalar_lanes"]


class AddressMappingKind(Enum):
    """How a kernel's DRAM-facing side (its first-stage load or its
    last-stage store) walks memory across the sub-FFT's own N elements.

    CONTIGUOUS: element index i sits at `row*row_stride + i` -- a unit-stride
    vector load/store, one instruction per SIMD batch.

    STRIDED: element index i sits at `row*row_stride + i*elem_stride` with
    `elem_stride != 1` -- there is no vector instruction for that, so codegen
    falls back to one scalar access per lane (the same fallback already used
    for a partial/tail SIMD batch).

    `row` is always `global_uthread_id()`: the logical sub-FFT id. This is
    address generation, not a physical shared-memory transpose -- see
    `AddressMapping`.

    SPLIT: element index i sits at `(row % a) + (row // a) * (a * kernel_length)
    + i * a` -- originally a middle kernel's output within an M>=3
    multi-kernel chain, whose `row` mixes an already-transformed digit run
    (weight < a) with a not-yet-transformed remainder (weight >= a).
    Superseded by PEELED (below) for that role -- make_multi_kernel_plan
    builds every M>=3 chain with PEELED now, and no `AddressMapping`
    constructor produces SPLIT any more. Kept only as the value PEELED's
    own docstring contrasts itself against. CONTIGUOUS is the a=1 special
    case of this same formula (the modulo/div both vanish) -- kept as its
    own kind because it needs neither a division nor the kernel's own
    length, and is the only kind M<=2 chains ever use.

    PEELED: a non-last kernel's output, chosen so the *next* kernel's own
    input read becomes `strided(row_stride=next kernel's length,
    elem_stride=1)` -- a real vector load -- instead of today's
    `strided(row_stride=1, elem_stride=n // next kernel's length)`, which
    is always scalar-per-lane. `row` mixes three things: an
    already-transformed run (weight < a, same `a` as SPLIT), the next
    kernel's own digit `d_next` (weight `tail_size`, within the
    not-yet-transformed remainder `row // a`), and everything still after
    that (`rest`, weight >= tail_size). SPLIT keeps the whole
    not-yet-transformed remainder together at the outer (large-stride) end;
    PEELED instead pulls `d_next` out to the innermost slot and pushes
    `rest` outward, so the *next* kernel's read costs it nothing. Element
    index i (this kernel's own new digit) sits between: `(row %
    a)*k_next + i*(a*k_next) + ((row // a) // tail_size) + ((row // a) %
    tail_size)*(a*kernel_length*k_next)`. Derived and verified by direct
    index-permutation simulation (bijective, and every downstream read
    pulls a clean single-digit sweep) up to a 10-kernel chain before this
    touched fft_plangen.py -- same discipline SPLIT's own derivation used.
    This is a read-locality optimization, not a stride bound: the worst
    stride anywhere in the chain is unchanged (conserved, not reduced) --
    it moves from every kernel's read to this kernel's own write, whose
    `i*(a*k_next)` term grows exactly the way SPLIT's old `i*a` did. What
    it does change, provably: every kernel after the first gets a real
    vector read instead of a forced scalar one.

    CROSSED: the last kernel of one *side* of a make_balanced_plan split
    (N = N_A * N_B, each side possibly its own multi-kernel PEELED chain --
    see make_balanced_plan), writing so the *other* side can start its own
    chain fresh (bounded by that side's own size, not by N). `row` mixes a
    batch index (weight 1, range `batch_count` -- which of the other
    side's independent transforms this uthread belongs to) with this
    side's own already-transformed digits (weight `batch_count`, the same
    kind of `a`-scoped quantity SPLIT/PEELED already track, just seeded to
    `batch_count` instead of 1 for this whole side's chain). Element index
    i (this kernel's own new digit) sits at `(row % batch_count) *
    side_length + (row // batch_count) + i * digit_multiplier`: batch
    outermost (so the far side's own first kernel can read this side's
    `side_length` elements contiguously), this side's fully-combined
    digit (`row // batch_count` combined with `i`) innermost. Degenerates
    to plain `contiguous(row_stride=side_length)` when this side is a
    single kernel (`digit_multiplier=1`, `row` already just the batch
    index) -- exactly make_decomposed_plan's own kernel0 formula, checked
    directly as the base case before trusting the multi-kernel
    generalization. Verified by direct index simulation *and* full
    complex-arithmetic comparison against numpy.fft (both sides
    multi-kernel chains, forward and inverse) before this touched
    fft_plangen.py.
    """

    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    SPLIT = "split"
    PEELED = "peeled"
    CROSSED = "crossed"


@dataclass(frozen=True)
class AddressMapping:
    """addr(row, elem) = base + row*row_stride + elem*elem_stride.

    `row` is this sub-FFT's logical id (`global_uthread_id()`, a runtime
    value -- one multiply). `elem` is the natural in-FFT element index
    (0..length-1), which `_lower_stages` already resolves to a plan-time
    constant per load/store; multiplying it by `elem_stride` stays a
    plan-time constant too. Nothing here is decided by codegen: a plan
    carries one `AddressMapping` for its DRAM input and one for its DRAM
    output, and codegen only ever reads `row_stride`/`elem_stride`/`base`
    off of them.

    A single-kernel FFT's input and output are both
    `AddressMapping.contiguous(length)` -- the plan's `batch_stride` today.
    A decomposed FFT's kernels use `strided(...)` on whichever side isn't
    naturally contiguous, so that the *other* side keeps its vector
    load/store. See `make_decomposed_plan` for how N0*N1 picks these.
    """

    kind: AddressMappingKind
    row_stride: int
    elem_stride: int = 1
    base: int = 0

    # PEELED only -- see AddressMappingKind.PEELED. row_stride is unused
    # (and left 0) for this kind: PEELED's row-dependent term isn't a
    # single multiply, so _mapping_base_expr dispatches on these instead.
    peel_a: int = 0
    peel_k_next: int = 0
    peel_tail_size: int = 0

    @staticmethod
    def contiguous(row_stride: int, base: int = 0) -> "AddressMapping":
        return AddressMapping(AddressMappingKind.CONTIGUOUS, row_stride, 1, base)

    @staticmethod
    def strided(row_stride: int, elem_stride: int, base: int = 0) -> "AddressMapping":
        if elem_stride == 1:
            raise ValueError("a strided mapping needs elem_stride != 1")
        return AddressMapping(AddressMappingKind.STRIDED, row_stride, elem_stride, base)

    @staticmethod
    def peeled(a: int, k_next: int, tail_size: int, base: int = 0) -> "AddressMapping":
        """`a`: accumulated product of kernels processed before this one in
        the chain. `k_next`: the *next* kernel's own local length.
        `tail_size`: product of every kernel's length after next (1 if
        there is none -- i.e. next is the last kernel in the chain).
        `a=1` is valid (this kernel is first in the chain).
        """
        if a <= 0:
            raise ValueError("a peeled mapping needs a > 0")
        if k_next <= 0 or tail_size <= 0:
            raise ValueError("a peeled mapping needs k_next > 0 and tail_size > 0")
        return AddressMapping(
            AddressMappingKind.PEELED,
            row_stride=0,
            elem_stride=a * k_next,
            base=base,
            peel_a=a,
            peel_k_next=k_next,
            peel_tail_size=tail_size,
        )

    @staticmethod
    def crossed(
        batch_count: int, side_length: int, digit_multiplier: int, base: int = 0
    ) -> "AddressMapping":
        """`batch_count`: how many independent transforms on the *other*
        side this side's own output is interleaved across. `side_length`:
        this side's own total length (N_A or N_B). `digit_multiplier`: the
        product of this side's own chunk lengths *before* the current
        (last) one -- 1 if this side is a single kernel. Reuses `peel_a`
        for `batch_count` and `row_stride` for `side_length`; see
        AddressMappingKind.CROSSED.
        """
        if batch_count <= 0 or side_length <= 0 or digit_multiplier <= 0:
            raise ValueError(
                "a crossed mapping needs batch_count, side_length, "
                "digit_multiplier all > 0"
            )
        return AddressMapping(
            AddressMappingKind.CROSSED,
            row_stride=side_length,
            elem_stride=digit_multiplier,
            base=base,
            peel_a=batch_count,
        )


@dataclass(frozen=True)
class ScratchpadBufferPlan:
    name: str
    elements: int


@dataclass(frozen=True)
class LoadPlan:
    """One already-resolved complex SIMD input load."""

    operand: int
    source: LoadSource
    buffer_name: str | None
    mode: LoadMode

    # Used by vector mode.
    base_offset: int | None = None

    # Used by scalar_pack mode.  One entry for every SIMD lane.
    # None means that lane is planner-selected padding and must become zero.
    packed_lane_offsets: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class TwiddlePlan:
    """Exact SIMD twiddle constants for one radix output."""

    real: tuple[float, ...]
    imag: tuple[float, ...]


@dataclass(frozen=True)
class StorePlan:
    """One already-resolved complex output store."""

    destination: StoreDestination
    buffer_name: str | None
    mode: StoreMode

    # Used by vector mode.
    base_offset: int | None = None

    # Used by scalar_lanes mode.  One exact destination offset per valid lane.
    lane_offsets: tuple[int, ...] = ()


@dataclass(frozen=True)
class LargeTwiddlePlan:
    """Cross-block twiddle for one factor of an N = N0*N1 decomposition,
    fused into a kernel's final-stage output path (never a separate stage
    or kernel -- see module docstring).

    W_N^(row*output), row = global_uthread_id() (this kernel's logical
    sub-FFT id, a *runtime* value) and output = this stage's plan-time-known
    output digit. That runtime dependency is exactly what makes this
    different from the ordinary per-stage `TwiddlePlan`: an exponent that
    depends on `global_uthread_id()` cannot be folded into a SIMD compile-time
    constant, so it cannot reuse that mechanism.

    twiddle source: a small DRAM table the host precomputes once at startup,
    with std.math cos/sin -- the same "computed at host runtime, not baked
    into the plan" approach `fft_codegen.py` already uses for the reference
    DFT. `full_length` (== the overall FFT's N, not a per-kernel value) is
    the table's own size, laid out to match this kernel's own
    `output_mapping` exactly -- `row_count*output_count == full_length //
    a` distinct values, but the table itself is `full_length` long and the
    fetch reuses the store's own already-computed address as-is (so a
    SPLIT output_mapping's address, which mixes an already-transformed `a`
    with this stage's `output`, reads a value that's the same across every
    `a` -- `a`-fold redundant, not a second independent address
    computation). `a=1` (the default, and every M<=2 chain's only case) is
    the M=2/kernel0 form: row_count*output_count == full_length exactly,
    no redundancy.

    A full twiddle table is the direct approach; it is what's needed here.
    rocFFT/clFFT-style digit-split reconstruction (W_N^u = T0[u0]*T1[u1]*...)
    would trade table size for extra multiplies on very large N -- worth
    adding if a table of size N stops being affordable, not before.

    `output_mapping` may be SPLIT or PEELED (see AddressMappingKind); the
    twiddle *value* (W_N^(row*output), i.e. row_count/output_count/a/
    inverse above) never depends on which one, since that's the same
    underlying math either way -- only the table's own physical layout
    does, since the fetch reuses the store's address as-is. `k_next` and
    `tail_size` (0/1 for SPLIT, matching AddressMapping.peeled's own
    parameters for PEELED) select which address formula fills the table.
    """

    full_length: int
    row_count: int
    output_count: int
    inverse: bool
    a: int = 1
    table_name: str = "large_twiddle"
    # 0 selects the legacy SPLIT table layout; a positive value selects
    # PEELED's (see AddressMapping.peeled -- same two parameters).
    k_next: int = 0
    tail_size: int = 1


@dataclass(frozen=True)
class OutputPlan:
    output: int
    twiddle: TwiddlePlan | None
    scale: float | None
    large_twiddle: bool
    store: StorePlan


@dataclass(frozen=True)
class SIMDBatchPlan:
    batch_id: int
    valid_lanes: int
    loads: tuple[LoadPlan, ...]
    outputs: tuple[OutputPlan, ...]


@dataclass(frozen=True)
class FFTStagePlan:
    stage_id: int
    radix: int
    inverse: bool
    simd_iteration_count: int
    batches: tuple[SIMDBatchPlan, ...]

    # Cooperative execution only (see CooperationPlan / fft_plan_cooperative.py):
    # `batches` above, partitioned across this stage's workers -- worker `w`
    # executes exactly `worker_batches[w]` (a subset of the same, already-fully-
    # resolved SIMDBatchPlan objects in `batches`; nothing about a batch's own
    # load/twiddle/store offsets changes depending on who executes it, since
    # those are already absolute positions within this FFT, not uthread-
    # relative -- see fft_plan_cooperative.py's module docstring). `None`
    # (every plan built by `_build_plan` directly) means today's behavior: one
    # implicit worker owns every batch in `batches`, unchanged. `len(worker_batches)`
    # is this stage's own `active_workers` -- a worker whose id is >= that
    # (or whose own entry is empty) does nothing this stage.
    worker_batches: tuple[tuple[SIMDBatchPlan, ...], ...] | None = None


@dataclass(frozen=True)
class CooperationPlan:
    """Attached to a leaf `FFTCodegenPlan` (see `FFTCodegenPlan.cooperation`)
    when that leaf is executed by more than one cooperating microthread per
    sub-FFT, instead of today's default "1 uthread = 1 whole sub-FFT" -- see
    fft_plan_cooperative.py for the builder and the full design rationale
    (scratchpad-capacity motivation, the local_uthread_id()-based fft_slot/
    worker_id split, the group_id()/num_groups()-based logical FFT id that
    replaces global_uthread_id() for a cooperative leaf's own DRAM mapping).

    `workers_per_fft`: how many microthreads cooperate on one sub-FFT (their
    local_uthread_id()s are consecutive: fft_slot = local_id // workers_per_fft,
    worker_id = local_id % workers_per_fft).

    `fft_slots_per_group`: how many independent sub-FFTs' worth of scratchpad
    fit on one NDP unit at once -- this is what `FFTCodegenPlan.max_uthread`
    meant for a non-cooperative plan; for a cooperative one, `max_uthread`
    instead holds the *physical* per-group microthread cap
    (`fft_slots_per_group * workers_per_fft`), since that is what actually
    sizes the launch / drives round-split, so `fft_slots_per_group` is kept
    here rather than overloading `max_uthread`'s meaning a second way.
    """

    workers_per_fft: int
    fft_slots_per_group: int


@dataclass(frozen=True)
class HostPlan:
    """Parameters for the host-generated test: `fft_codegen.py` emits Mojo
    code that generates its own random input and its own reference at
    *runtime* (std.random + a direct O(N^2) DFT, independent of the radix
    decomposition the kernel runs) -- there are no baked-in numbers to plan
    here, only the shape of that runtime computation."""

    total_elems: int
    pool_elems: int
    length: int
    total_uthreads: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class FFTCodegenPlan:
    """Fully lowered plan consumed by fft_codegen.py.

    One `FFTCodegenPlan` is one *kernel*: one `NDPTask` struct, one launch,
    one `local_uthread_id()`-gated scratchpad region. A single-kernel FFT is
    exactly one of these. A decomposed FFT (see `DecomposedFFTPlan`) is two,
    chained through DRAM the way `two_tasks.mojo` chains `Scale`/`AddB` --
    never through a shared scratchpad, since nothing is guaranteed still
    resident once a kernel launch returns.

    `total_uthreads` and `max_uthread` are two different quantities that
    the field name `max_uthread` alone used to carry, ambiguously:

    * `total_uthreads` -- how many microthreads this kernel's *launch*
      covers in total (`PooledRange.over(pool, simd_lanes*total_uthreads)`),
      spread across however many NDP units the hardware maps them onto.
    * `max_uthread` -- how many of *one NDP unit's own* microthreads
      `local_uthread_id()` ever ranges over (docs/PRIMITIVES.md's
      `spad_capacity()`, "bytes of scratchpad on one unit", divided by
      what one uthread's own scratchpad footprint costs -- see
      `_build_plan`'s `spad_capacity_bytes`). This is what
      `ScratchpadBufferPlan.elements` and the `MAX_UTHREAD_<kernel>`
      device-side guard are sized against, since the scratchpad is one
      instance *per core*, shared by every uthread that lands on it --
      not one slot per uthread in the whole launch.

    `max_uthread <= total_uthreads` always; they're equal (today's
    behavior, unchanged) whenever `_build_plan` isn't given a
    `spad_capacity_bytes` to size against, or the kernel uses no
    scratchpad at all (a single-stage kernel -- see
    `_scratchpad_buffer_names`), since then there's nothing to divide
    across units.
    """

    length: int
    inverse: bool
    total_uthreads: int
    max_uthread: int
    simd_lanes: int
    kernel_name: str

    # DRAM-facing address mappings -- see `AddressMapping`. A single-kernel
    # FFT's are both `contiguous(length)`, which is today's `batch_stride`
    # generalized: row_stride==length, elem_stride==1 on both sides.
    input_mapping: AddressMapping
    output_mapping: AddressMapping
    large_twiddle: LargeTwiddlePlan | None

    scratchpad_uthread_stride: int

    scratchpad_buffers: tuple[ScratchpadBufferPlan, ...]
    stages: tuple[FFTStagePlan, ...]
    host: HostPlan

    # None (every plan built directly by `_build_plan`): today's behavior,
    # one uthread per whole sub-FFT, unchanged. See `CooperationPlan` /
    # fft_plan_cooperative.py.
    cooperation: CooperationPlan | None = None


@dataclass(frozen=True)
class _StageLayout:
    """Planner-internal stage description.

    These fields are intentionally not exposed to fft_codegen.py.  The planner
    lowers them into explicit load/twiddle/store operations first.
    """

    radix: int
    butterfly_count: int
    input_batch_width: int
    input_stride: int
    output_batch_width: int
    output_stride: int
    twiddle_modulus: int | None = None
    twiddle_stride: int = 1
    twiddle_lane_divisor: int = 1


def _check_layouts(
    *,
    length: int,
    total_uthreads: int,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
) -> None:
    if length <= 0:
        raise ValueError("FFT length must be positive")
    if total_uthreads <= 0:
        raise ValueError("total_uthreads must be positive")
    if simd_lanes <= 0:
        raise ValueError("simd_lanes must be positive")
    if not layouts:
        raise ValueError("at least one FFT stage is required")

    product = 1
    for sid, stage in enumerate(layouts):
        if stage.radix not in SUPPORTED_RADICES:
            supported = ", ".join(str(r) for r in sorted(SUPPORTED_RADICES))
            raise ValueError(
                f"stage {sid}: radix-{stage.radix} is unsupported; "
                f"supported radices are {{{supported}}}"
            )
        if stage.butterfly_count <= 0:
            raise ValueError(f"stage {sid}: butterfly_count must be positive")
        if stage.input_batch_width <= 0 or stage.output_batch_width <= 0:
            raise ValueError(f"stage {sid}: batch width must be positive")
        if stage.input_stride <= 0 or stage.output_stride <= 0:
            raise ValueError(f"stage {sid}: stride must be positive")
        if stage.twiddle_modulus is not None and stage.twiddle_modulus <= 0:
            raise ValueError(f"stage {sid}: twiddle_modulus must be positive")
        if stage.twiddle_lane_divisor <= 0:
            raise ValueError(f"stage {sid}: twiddle_lane_divisor must be positive")
        product *= stage.radix

    if product != length:
        raise ValueError(
            f"product of stage radices ({product}) must equal FFT length ({length})"
        )


def _cap_max_uthread(
    total_uthreads: int,
    bytes_per_uthread: int,
    spad_capacity_bytes: int | None,
    *,
    context: str,
    max_concurrent_scratchpad_bytes: int | None = None,
) -> int:
    """How many of `total_uthreads` fit on one NDP unit's own scratchpad,
    given `bytes_per_uthread` each need -- shared by every kernel/tile
    builder in this module that sizes a scratchpad-backed launch against
    an optional per-unit capacity (see FFTCodegenPlan's own docstring for
    total_uthreads vs. max_uthread). `bytes_per_uthread <= 0` (nothing
    scratchpad-backed to cap against) or `spad_capacity_bytes is None`
    (no capacity given) both mean "no byte cap" -- the always-safe,
    possibly oversized default every caller used before this was factored
    out.

    `max_concurrent_scratchpad_bytes`: a second, independent cap on how
    many bytes of scratchpad may be *concurrently active* across every
    uthread resident on one unit at once (`max_uthread * bytes_per_uthread`)
    -- `None` (the default) applies none. Found empirically on N=8192's
    FFTRecNear0 (2048 bytes/uthread): 16 concurrent uthreads (32768 bytes)
    finishes in ~90K simulated cycles; 30 (61440 bytes) blew *past* the
    simulator's 20,000,000-cycle budget for the same kernel and data. A
    flat *count* cap (e.g. always 16) was the first fix tried, and it does
    stop that -- but it then wrongly re-caps small-footprint kernels that
    were never at risk (N=1024's FFTRecLeaf1, 32 bytes/uthread, ran 256
    concurrent uthreads -- 8192 bytes total, well under budget -- in ~1.1K
    cycles uncapped; forced down to 16 uthreads it still finishes, just
    across many more, needlessly small launches, each paying its own
    per-launch overhead for no reason). Capping the *byte product* instead
    scales the allowed uthread count down only for kernels whose own
    per-uthread footprint would actually approach the same contention,
    leaving small-footprint kernels uncapped.
    """
    if bytes_per_uthread <= 0 or spad_capacity_bytes is None:
        result = total_uthreads
    elif bytes_per_uthread > spad_capacity_bytes:
        raise ValueError(
            f"spad_capacity_bytes={spad_capacity_bytes} is too small to fit "
            f"even a single uthread of {context} (needs {bytes_per_uthread} bytes)"
        )
    else:
        result = min(total_uthreads, spad_capacity_bytes // bytes_per_uthread)
    if max_concurrent_scratchpad_bytes is not None and bytes_per_uthread > 0:
        result = min(result, max_concurrent_scratchpad_bytes // bytes_per_uthread)
    return result


def pingpong_needed(stage_count: int) -> bool:
    """Whether a `stage_count`-stage kernel actually needs two ping-pong
    scratchpad banks, vs. one shared buffer that halves scratchpad usage.

    Only a *middle* stage of a Stockham chain both reads and writes its own
    kernel's scratchpad in the same stage: stage 0 only ever reads DRAM and
    writes scratchpad, and the last stage only ever reads scratchpad and
    writes DRAM (see `_make_load`/`_make_store`'s own `first_stage`/
    `last_stage` source/destination split) -- so a chain of exactly 1 or 2
    stages has no stage that both reads and writes the buffer at once, and
    one shared buffer is already race-free. Proven directly, not just
    argued: numerically verified across many radix combinations at depth 1
    and 2 via `verify_fft_simple.verify_radix_sequence_plan`'s own sweep
    (every `radix_sequence_cases` entry of length <=2 in verify_fft_plan.py's
    `main()` now exercises the single-buffer path for real), and confirmed
    by hand that forcing single-buffer on a 3-stage chain instead
    (`(4,4,4)`, `(2,2,2,2,2,2)`) corrupts the output (max error ~10-40,
    vs. ~1e-8 for every passing case) before this helper existed.

    3+ stages always need both banks: a middle stage's own read (the
    previous stage's output) and write (the next stage's input) cannot
    share one buffer without risking a same-stage read-after-write, since
    the Stockham store permutation may scatter an early SIMD batch's output
    onto an address a later, not-yet-processed batch still needs to read.
    """
    return stage_count > 2


def _scratchpad_buffer_names(
    *,
    stage_count: int,
    use_pingpong: bool,
) -> tuple[str, ...]:
    # No intermediate result exists for a one-stage FFT.
    if stage_count <= 1:
        return ()
    if use_pingpong:
        return ("buf_a", "buf_b")
    return ("buf",)


def _write_buffer_for_stage(
    *,
    stage_id: int,
    stage_count: int,
    buffer_names: tuple[str, ...],
) -> str | None:
    if stage_id == stage_count - 1:
        return None
    if not buffer_names:
        raise ValueError("intermediate stage requires scratchpad storage")
    if len(buffer_names) == 1:
        return buffer_names[0]
    return buffer_names[stage_id % len(buffer_names)]


def _read_buffer_for_stage(
    *,
    stage_id: int,
    stage_count: int,
    buffer_names: tuple[str, ...],
) -> str | None:
    if stage_id == 0:
        return None
    return _write_buffer_for_stage(
        stage_id=stage_id - 1,
        stage_count=stage_count,
        buffer_names=buffer_names,
    )


def _make_load(
    *,
    operand: int,
    first_stage: bool,
    read_buffer: str | None,
    base_offset: int,
    valid_lanes: int,
    simd_lanes: int,
    dram_elem_stride: int = 1,
) -> LoadPlan:
    source: LoadSource = "input" if first_stage else "scratchpad"
    buffer_name = None if first_stage else read_buffer

    if source == "scratchpad" and buffer_name is None:
        raise ValueError("scratchpad load requires a resolved buffer name")

    # A non-unit DRAM element stride (this kernel's input_mapping is
    # STRIDED -- see AddressMapping) has no vector-load form: fall back to
    # one scalar load per lane, same fallback a partial/tail SIMD batch
    # already uses. Only the DRAM ("input") side can be strided this way;
    # scratchpad loads are always this kernel's own contiguous layout.
    force_scalar = source == "input" and dram_elem_stride != 1

    if valid_lanes == simd_lanes and not force_scalar:
        return LoadPlan(
            operand=operand,
            source=source,
            buffer_name=buffer_name,
            mode="vector",
            base_offset=base_offset,
        )

    stride = dram_elem_stride if source == "input" else 1
    return LoadPlan(
        operand=operand,
        source=source,
        buffer_name=buffer_name,
        mode="scalar_pack",
        packed_lane_offsets=tuple(
            ((base_offset + lane) * stride) if lane < valid_lanes else None
            for lane in range(simd_lanes)
        ),
    )


def _make_twiddle(
    *,
    inverse: bool,
    simd_lanes: int,
    valid_lanes: int,
    first_bfly: int,
    output: int,
    modulus: int | None,
    stride: int,
    lane_divisor: int,
) -> TwiddlePlan | None:
    if modulus is None or output == 0:
        return None

    sign = 1.0 if inverse else -1.0
    wr: list[float] = []
    wi: list[float] = []
    for lane in range(simd_lanes):
        if lane >= valid_lanes:
            wr.append(1.0)
            wi.append(0.0)
            continue

        bfly = first_bfly + lane
        twiddle_index = bfly // lane_divisor
        exponent = twiddle_index * output * stride
        angle = sign * 2.0 * pi * exponent / modulus
        wr.append(cos(angle))
        wi.append(sin(angle))

    return TwiddlePlan(real=tuple(wr), imag=tuple(wi))


def _make_store(
    *,
    last_stage: bool,
    write_buffer: str | None,
    layout: _StageLayout,
    simd_it: int,
    output: int,
    valid_lanes: int,
    simd_lanes: int,
    dram_elem_stride: int = 1,
) -> StorePlan:
    output_batch_base = simd_it * layout.output_batch_width

    if last_stage:
        base = (output_batch_base + output * layout.output_stride) * dram_elem_stride
        # Symmetric with _make_load: a non-unit DRAM element stride (this
        # kernel's output_mapping is STRIDED) has no vector-store form.
        if valid_lanes == simd_lanes and dram_elem_stride == 1:
            return StorePlan(
                destination="output",
                buffer_name=None,
                mode="vector",
                base_offset=base,
            )
        return StorePlan(
            destination="output",
            buffer_name=None,
            mode="scalar_lanes",
            lane_offsets=tuple(
                base + lane * dram_elem_stride for lane in range(valid_lanes)
            ),
        )

    if write_buffer is None:
        raise ValueError("intermediate stage requires a resolved write buffer")

    # One formula for every non-last (Stockham-autosort) stage. P_s --
    # this stage's twiddle_lane_divisor, the product of the radices before
    # it -- picks both the twiddle exponent (in _make_twiddle) and this
    # store permutation, so the next stage's read is always the same
    # "N_local/radix contiguous blocks" shape (see _make_load's
    # input_stride) regardless of which stage wrote it or what radix it
    # was. Was two hand-authored special cases (one per stage position);
    # collapsed after confirming both were this same formula evaluated at
    # P_s=1 and P_s=radix respectively.
    p_s = layout.twiddle_lane_divisor
    group_stride = layout.radix * p_s
    offsets = []
    for lane in range(valid_lanes):
        bfly = simd_it * simd_lanes + lane
        n2 = bfly // p_s
        b_s = bfly % p_s
        offsets.append(n2 * group_stride + output * p_s + b_s)

    # Store vectorization: `offsets` is already exact (every lane's real
    # destination, computed above) -- whether consecutive lanes land on
    # consecutive addresses is a fact about *this* batch's own offsets,
    # not a new layout decision, so promoting to a single vector store
    # whenever that fact holds is still purely this function's job, same
    # as the last-stage branch above already does for the DRAM case.
    # `p_s >= simd_lanes` is when this is true in practice (this stage's
    # cumulative radix product has grown past one SIMD batch, so `n2`
    # stays constant across the whole batch and `b_s` alone walks
    # consecutively) -- checked directly here rather than trusted, so nothing
    # downstream has to know why. Full-width only (`valid_lanes ==
    # simd_lanes`): a tail batch keeps the always-safe scalar_lanes path,
    # exactly the "otherwise -> scalar_lanes" fallback fft_codegen.py's own
    # _chunk_store leans on for compute_lanes-sized sub-slices later.
    if valid_lanes == simd_lanes and all(
        offsets[i] == offsets[0] + i for i in range(1, valid_lanes)
    ):
        return StorePlan(
            destination="scratchpad",
            buffer_name=write_buffer,
            mode="vector",
            base_offset=offsets[0],
        )
    return StorePlan(
        destination="scratchpad",
        buffer_name=write_buffer,
        mode="scalar_lanes",
        lane_offsets=tuple(offsets),
    )


def _lower_stages(
    *,
    length: int,
    inverse: bool,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
    buffer_names: tuple[str, ...],
    input_mapping: AddressMapping,
    output_mapping: AddressMapping,
    large_twiddle: LargeTwiddlePlan | None,
    inverse_scale: float | None,
) -> tuple[FFTStagePlan, ...]:
    stages: list[FFTStagePlan] = []
    stage_count = len(layouts)

    for stage_id, layout in enumerate(layouts):
        first_stage = stage_id == 0
        last_stage = stage_id == stage_count - 1
        read_buffer = _read_buffer_for_stage(
            stage_id=stage_id,
            stage_count=stage_count,
            buffer_names=buffer_names,
        )
        write_buffer = _write_buffer_for_stage(
            stage_id=stage_id,
            stage_count=stage_count,
            buffer_names=buffer_names,
        )

        simd_iters = (layout.butterfly_count + simd_lanes - 1) // simd_lanes
        batches: list[SIMDBatchPlan] = []

        for simd_it in range(simd_iters):
            first_bfly = simd_it * simd_lanes
            valid_lanes = min(simd_lanes, layout.butterfly_count - first_bfly)
            input_batch_base = simd_it * layout.input_batch_width

            loads = tuple(
                _make_load(
                    operand=operand,
                    first_stage=first_stage,
                    read_buffer=read_buffer,
                    base_offset=input_batch_base + operand * layout.input_stride,
                    valid_lanes=valid_lanes,
                    simd_lanes=simd_lanes,
                    dram_elem_stride=(
                        input_mapping.elem_stride if first_stage else 1
                    ),
                )
                for operand in range(layout.radix)
            )

            outputs: list[OutputPlan] = []
            for output in range(layout.radix):
                outputs.append(
                    OutputPlan(
                        output=output,
                        twiddle=_make_twiddle(
                            inverse=inverse,
                            simd_lanes=simd_lanes,
                            valid_lanes=valid_lanes,
                            first_bfly=first_bfly,
                            output=output,
                            modulus=layout.twiddle_modulus,
                            stride=layout.twiddle_stride,
                            lane_divisor=layout.twiddle_lane_divisor,
                        ),
                        scale=inverse_scale if last_stage else None,
                        large_twiddle=last_stage and large_twiddle is not None,
                        store=_make_store(
                            last_stage=last_stage,
                            write_buffer=write_buffer,
                            layout=layout,
                            simd_it=simd_it,
                            output=output,
                            valid_lanes=valid_lanes,
                            simd_lanes=simd_lanes,
                            dram_elem_stride=(
                                output_mapping.elem_stride if last_stage else 1
                            ),
                        ),
                    )
                )

            batches.append(
                SIMDBatchPlan(
                    batch_id=simd_it,
                    valid_lanes=valid_lanes,
                    loads=loads,
                    outputs=tuple(outputs),
                )
            )

        stages.append(
            FFTStagePlan(
                stage_id=stage_id,
                radix=layout.radix,
                inverse=inverse,
                simd_iteration_count=simd_iters,
                batches=tuple(batches),
            )
        )

    return tuple(stages)


def _make_host_plan(
    *,
    length: int,
    inverse: bool,
    total_uthreads: int,
    simd_lanes: int,
) -> HostPlan:
    """No numbers are baked in here: `fft_codegen.py` emits Mojo that
    generates its own random input (`std.random`, seeded, matching every
    other benchmark's `main()`) and its own reference (a direct O(N^2) DFT
    computed at host runtime, independent of the radix decomposition the
    kernel runs) each time the program is launched, rather than a fixed
    signal computed once by this planner and frozen into the source. See
    `fft_codegen.py`'s host-emission section for that computation."""
    return HostPlan(
        total_elems=length * total_uthreads,
        pool_elems=simd_lanes * total_uthreads,
        length=length,
        total_uthreads=total_uthreads,
        inverse=inverse,
        tolerance=1.0e-3,
    )


class _Default:
    """Sentinel distinguishing 'caller didn't say' from 'caller explicitly
    said None' -- see `_build_plan`'s `inverse_scale`."""


_DEFAULT = _Default()


def _build_plan(
    *,
    length: int,
    inverse: bool,
    total_uthreads: int,
    simd_lanes: int,
    use_pingpong: bool,
    layouts: tuple[_StageLayout, ...],
    kernel_name: str = "FFTFP32",
    input_mapping: AddressMapping | None = None,
    output_mapping: AddressMapping | None = None,
    large_twiddle: LargeTwiddlePlan | None = None,
    inverse_scale: float | None | _Default = _DEFAULT,
    spad_capacity_bytes: int | None = None,
    max_concurrent_scratchpad_bytes: int | None = None,
) -> FFTCodegenPlan:
    """`spad_capacity_bytes` is one NDP unit's own scratchpad size (see
    `FFTCodegenPlan`'s docstring) -- optional and `None` by default, which
    keeps today's behavior exactly (`max_uthread == total_uthreads`, i.e.
    assume the whole launch could land on one unit, the always-safe but
    possibly oversized choice `elements = scratchpad_stride * total_uthreads`
    already made). Given a real budget, `max_uthread` is capped to however
    many of *this* kernel's own uthreads (each needing
    `len(buffer_names) * scratchpad_stride * 4` bytes for its ping-pong
    footprint) fit in it -- moot for a single-stage kernel, which uses no
    scratchpad at all (see `_scratchpad_buffer_names`).
    """
    _check_layouts(
        length=length,
        total_uthreads=total_uthreads,
        simd_lanes=simd_lanes,
        layouts=layouts,
    )

    # Default: today's single-kernel behavior -- one contiguous row per
    # sub-FFT, `row_stride == length`, unchanged from the old hardcoded
    # `batch_stride`.
    if input_mapping is None:
        input_mapping = AddressMapping.contiguous(length)
    if output_mapping is None:
        output_mapping = AddressMapping.contiguous(length)
    if inverse_scale is _DEFAULT:
        inverse_scale = (1.0 / length) if inverse else None

    buffer_names = _scratchpad_buffer_names(
        stage_count=len(layouts),
        use_pingpong=use_pingpong,
    )
    scratchpad_stride = 2 * length

    bytes_per_uthread = len(buffer_names) * scratchpad_stride * 4 if buffer_names else 0
    max_uthread = _cap_max_uthread(
        total_uthreads, bytes_per_uthread, spad_capacity_bytes,
        context=f"kernel {kernel_name!r}",
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )

    scratchpad_buffers = tuple(
        ScratchpadBufferPlan(
            name=name,
            elements=scratchpad_stride * max_uthread,
        )
        for name in buffer_names
    )

    return FFTCodegenPlan(
        length=length,
        inverse=inverse,
        total_uthreads=total_uthreads,
        max_uthread=max_uthread,
        simd_lanes=simd_lanes,
        kernel_name=kernel_name,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        large_twiddle=large_twiddle,
        scratchpad_uthread_stride=scratchpad_stride,
        scratchpad_buffers=scratchpad_buffers,
        stages=_lower_stages(
            length=length,
            inverse=inverse,
            simd_lanes=simd_lanes,
            layouts=layouts,
            buffer_names=buffer_names,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            large_twiddle=large_twiddle,
            inverse_scale=inverse_scale,
        ),
        host=_make_host_plan(
            length=length,
            inverse=inverse,
            total_uthreads=total_uthreads,
            simd_lanes=simd_lanes,
        ),
    )


def layouts_for_radices(
    length: int, radices: tuple[int, ...], simd_lanes: int
) -> tuple[_StageLayout, ...]:
    """Every stage's _StageLayout for a length-`length` FFT run as one
    kernel's `radices` sequence (in Cooley-Tukey/Stockham order: `length`
    == product(radices)), derived from nothing but the radix sequence
    itself -- no per-stage hand authoring.

    `P_s` (this stage's `twiddle_lane_divisor`) is the product of the
    radices before it; both `_make_twiddle`'s exponent and `_make_store`'s
    intermediate-stage permutation key off it (see `_make_store`), which
    is what makes every non-last stage's read of its input always the same
    "length/radix contiguous blocks" shape regardless of which stage wrote
    it (`input_stride = length // radix` here, on every stage).

    `input_batch_width`/`output_batch_width` are pinned to `simd_lanes`:
    they only matter (`simd_it * width`) once a stage has more than one
    SIMD batch, and at that point the load/store math requires the batch
    stride to be exactly the lane count.
    """
    stage_count = len(radices)
    layouts: list[_StageLayout] = []
    p = 1
    for stage_id, radix in enumerate(radices):
        last_stage = stage_id == stage_count - 1
        layouts.append(
            _StageLayout(
                radix=radix,
                butterfly_count=length // radix,
                input_batch_width=simd_lanes,
                input_stride=length // radix,
                output_batch_width=simd_lanes,
                output_stride=p if last_stage else 1,
                twiddle_modulus=None if last_stage else length // p,
                twiddle_stride=1,
                # Unused (and not necessarily a divisor of simd_lanes) on
                # the last stage: there is no twiddle there (modulus=None
                # above short-circuits _make_twiddle), and _make_store's
                # last-stage branch doesn't read this field either.
                twiddle_lane_divisor=1 if last_stage else p,
            )
        )
        p *= radix
    return tuple(layouts)


# ---------------------------------------------------------- number theory
#
# Shared by every planning strategy in this package that needs to factor N
# into a radix sequence (multi-kernel, balanced, balanced-transpose,
# recursive).


def _prime_factors_supported(n: int) -> list[int]:
    """Factor n into primes SUPPORTED_RADICES covers. Composite-radix
    coalescing (e.g. four radix-2 stages -> one radix-16) is deferred --
    see plan Stage 6 -- so only the prime subset of SUPPORTED_RADICES is
    used here, even though composites like 16 are themselves valid radices
    elsewhere in this module.
    """
    primes = sorted(r for r in SUPPORTED_RADICES if all(r % d for d in range(2, r)))
    factors: list[int] = []
    remaining = n
    for p in primes:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
    if remaining != 1:
        raise ValueError(
            f"{n} has a prime factor outside the primes SUPPORTED_RADICES "
            f"covers ({primes}); cannot factor for kernel chunking"
        )
    return factors


#  Confirmed via the real Mojo -> llc -> M2NDP-Detour toolchain (not just
# this module's own reasoning about register counts): radix-4 pairs
# (2,2)->4 are clean at every N tried, including deep chains (N=256, 1024:
# (4,4,4,4), zero spill warnings). radix-6 and radix-9 alone (N=6, N=9 --
# a radix that's *also* the kernel's only stage) are also clean, but *not*
# once that same radix has to read its operands from scratchpad instead of
# DRAM (any non-first stage): N=54 = (6,9) spills FFTRecLeaf0.stage_1 (the
# radix-9 stage, reading 9 complex operands out of scratchpad) to a
# `vs1r.v` the simulator doesn't implement -- the exact all-zero-output
# failure this project has hit before (see the loop_stages/round-split
# commit). radix-10 fails outright: N=160/320 = (4,4,10) spills
# FFTRecLeaf0/FFTRecNear0's own radix-10 stage the same way. So only
# radix-4 is confirmed safe as an *automatic*, always-on default -- 6/9/10
# stay available via `allowed` for a caller who has separately confirmed
# their own case doesn't spill, never wired into any call site by default.
_DEFAULT_COALESCE_ALLOWED: frozenset[int] = frozenset({4})


def coalesce_radices(
    factors: list[int] | tuple[int, ...], *, allowed: frozenset[int] | None = None
) -> tuple[int, ...]:
    """Merge adjacent pairs of `factors` (as `_prime_factors_supported`
    returns them: ascending prime value, equal primes grouped
    consecutively) into one larger SUPPORTED_RADICES composite wherever
    that pair's own product is itself allowed to merge into (see
    `_DEFAULT_COALESCE_ALLOWED` just above for why the default is
    radix-4-only, not "whatever SUPPORTED_RADICES contains") -- undoing
    _prime_factors_supported's own "composite-radix coalescing is
    deferred" simplification, now that a caller wants fewer, coarser
    stages instead of the maximal all-prime decomposition.

    Deliberately conservative and simple, per this project's own history
    with register pressure (see make_fft_kernel.py's `compute_lanes`
    docstring, and the loop_stages/round-split commit before this one):
    only ever merges *two* original factors at a time, greedily, left to
    right, and never re-merges an already-coalesced result with its
    neighbor. Every merged pair stays adjacent in the original Stockham/
    Cooley-Tukey factor order -- nothing here reorders, so this changes
    how many stages a decomposition renders as, never the decomposition
    itself.

    `allowed`: which composite products a pair may merge into -- `None`
    (the default) uses `_DEFAULT_COALESCE_ALLOWED` (radix-4 only, the one
    confirmed safe on real hardware in every configuration tried so far).
    A wider set (up to `SUPPORTED_RADICES` itself, which is where 6/9/10
    -- the other two-prime products that land back in SUPPORTED_RADICES,
    since reaching radix-8/16 needs a *triple* merge this pairs-only pass
    never attempts -- would come from) is available to a caller who has
    separately confirmed it doesn't spill for their own case, without
    touching this function. Exists so a future benchmark-driven cost
    model can swap in its own set (per-radix-cost-weighted, or simply
    wider once more of SUPPORTED_RADICES is confirmed spill-free) without
    touching any call site -- same "candidate-generation stays swappable"
    discipline _choose_recursive_split's own docstring already follows
    for its split-point search.
    """
    allowed_set = _DEFAULT_COALESCE_ALLOWED if allowed is None else allowed
    result: list[int] = []
    i = 0
    n = len(factors)
    while i < n:
        if i + 1 < n and factors[i] * factors[i + 1] in allowed_set:
            result.append(factors[i] * factors[i + 1])
            i += 2
        else:
            result.append(factors[i])
            i += 1
    return tuple(result)


# --------------------------------------------- shared M-kernel container
#
# MultiKernelFFTPlan/MultiKernelHostPlan are produced by more than one
# strategy (make_multi_kernel_plan, make_balanced_plan) so they live here,
# not in fft_plan_multikernel.py, alongside their own stride-summary tools.


@dataclass(frozen=True)
class MultiKernelHostPlan:
    n: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class MultiKernelFFTPlan:
    """N run as a chain of M>=1 kernels (see factor_into_kernel_chunks /
    make_multi_kernel_plan), each itself a layouts_for_radices multi-stage
    FFT. M=1 is exactly a single-kernel plan. M=2 no longer matches
    make_decomposed_plan's own addressing field-for-field (that function
    is untouched, still SPLIT/contiguous-based); this one's non-last
    kernels use AddressMappingKind.PEELED instead, chosen so every kernel
    after the first gets a vector (not scalar) DRAM read -- both are
    independently numpy-verified correct, they just lay the intermediate
    array out differently. See AddressMappingKind.PEELED.
    """

    n: int
    inverse: bool
    kernels: tuple[FFTCodegenPlan, ...]
    host: MultiKernelHostPlan


@dataclass(frozen=True)
class KernelStrideSummary:
    """Everything about one kernel's DRAM-facing layout that a plan summary
    or a stride-bound check needs -- derived entirely from fields
    FFTCodegenPlan already carries (input_mapping/output_mapping.elem_stride,
    length, each stage's radix), not new decisions. See
    summarize_multi_kernel_plan.
    """

    kernel_name: str
    local_length: int
    radices: tuple[int, ...]
    read_stride: int
    write_stride: int
    fused_transition: str  # "NONE" (this kernel's DRAM side is contiguous)
    #                         or "STORE" (a layout transition is folded into
    #                         this kernel's own output mapping -- see
    #                         AddressMappingKind.SPLIT / make_multi_kernel_plan;
    #                         this codebase never uses a LOAD-side fusion or a
    #                         standalone transpose kernel, so those aren't
    #                         separate enum values here, only documented cases
    #                         that don't occur -- see the plan's design notes).


def summarize_multi_kernel_plan(
    plan: MultiKernelFFTPlan,
) -> tuple[KernelStrideSummary, ...]:
    summaries = []
    for kernel in plan.kernels:
        transition = "NONE" if kernel.output_mapping.elem_stride == 1 else "STORE"
        summaries.append(
            KernelStrideSummary(
                kernel_name=kernel.kernel_name,
                local_length=kernel.length,
                radices=tuple(stage.radix for stage in kernel.stages),
                read_stride=kernel.input_mapping.elem_stride,
                write_stride=kernel.output_mapping.elem_stride,
                fused_transition=transition,
            )
        )
    return tuple(summaries)


def max_effective_stride(summaries: tuple[KernelStrideSummary, ...]) -> int:
    return max(
        max(s.read_stride, s.write_stride) for s in summaries
    )
