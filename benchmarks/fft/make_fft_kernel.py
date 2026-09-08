from __future__ import annotations

"""Generate a single, working M2NDP FFT Mojo kernel from nothing but a
length -- the same way every other benchmark in this repo is one
self-contained `.mojo` file (see scripts/build.sh's own comment: "A
benchmark is one file -- kernels, device_main and the host main that
launches them").

    python3 make_fft_kernel.py 960
    python3 make_fft_kernel.py 65536 --inverse -o out.mojo

This is a thin, deliberately boring wrapper around planning/fft_plan_recursive
.make_recursive_transpose_plan + codegen/fft_transpose_codegen.
generate_recursive_fft_kernels -- both already extensively verified (see
verification/) -- not a new implementation. That one planning strategy was
picked as the sole backend here because it is the only one that handles
*any* N on its own: it degenerates to a single fused kernel when N already
fits one uthread's own scratchpad, and recurses into a tiled-transpose
boundary only when it doesn't, so "just give it a number" always produces
something. The other strategies (make_444_plan, make_decomposed_plan,
make_multi_kernel_plan, make_balanced_plan, make_balanced_transpose_plan)
stay put in planning/ as the regression baseline/numeric reference they've
always been -- this file does not replace them, it's a convenience front
end for the one strategy general enough to be a sensible default.

The generated file's own shape is meant to be readable at a glance:
every kernel implementation first (one Mojo `struct` per stage -- see
generate_recursive_fft_kernels's own "KERNEL IMPLEMENTATIONS" banner),
then exactly one host `def main()` that allocates buffers, launches them
in order, and checks the result against an independent reference DFT
computed at runtime.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.search.fft_cost_model import estimate_cost, estimate_metrics
from planning.strategies.fft_plan_recursive import (
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
    make_recursive_transpose_plan,
)
from planning.search.fft_plan_search import (
    FFTPlanCandidate,
    PlanChoices,
    format_plan_summary,
    generate_candidates,
    rank_candidates,
)
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile


def _no_tail_tile_suggestions(plan: RecursiveFFTPlan) -> str:
    """One line per distinct PRE/MIDDLE/POST transpose shape in `plan`,
    listing every tile size that evenly divides both its `rows` and `cols`
    (every divisor of `gcd(rows, cols)`, square only -- see fft_plan_search.
    generate_tile_candidates' own comment for why exactly these tiles are
    the only ones real hardware has confirmed structurally spill-free at
    the tile level: no row/col tail means no LLVM tail-branch value
    specialization, the mechanism 2026-08-27's N=630 disassembly traced
    the actual spill to). Purely static -- no extra build+run -- so this
    is always cheap to compute and offer, unlike actually probing more
    candidates.

    Returns "" if `plan` has no transpose stage at all (a single fused
    leaf, nothing to tile).
    """
    import math

    seen: set[tuple[int, int]] = set()
    lines = []
    for node in flatten_recursive_node(plan.root):
        if not isinstance(node, PhysicalTransposePlan):
            continue
        shape = (node.rows, node.cols)
        if shape in seen:
            continue
        seen.add(shape)
        g = math.gcd(node.rows, node.cols)
        divisors = [d for d in range(1, g + 1) if g % d == 0]
        lines.append(
            f"    rows={node.rows} cols={node.cols}: no-tail tile sizes "
            f"{', '.join(f'{d}x{d}' for d in divisors)} (gcd={g}) -- confirmed "
            f"structurally spill-free by construction, though not guaranteed "
            f"fastest; still needs its own --verify-spill-free to confirm this "
            f"exact N/compute_lanes combination"
        )
    return "\n".join(lines)


GPU_BASELINE_PLANNERS = (
    "gpu-clfft",
    # "gpu-rocfft" is kept as the pre-existing name for the offline-tuner
    # port (see planning.gpu_baseline.rocfft's own module docstring);
    # "gpu-rocfft-tuned" is an explicit alias for the same thing, added
    # once the Phase-4 baseline audit found that ordinary rocFFT plan
    # creation runs a MATERIALLY DIFFERENT mechanism -- see "gpu-rocfft-
    # default" (planning.gpu_baseline.rocfft_default), which is NOT an
    # alias, it is a separate module porting that separate mechanism.
    "gpu-rocfft",
    "gpu-rocfft-tuned",
    "gpu-rocfft-default",
    "gpu-vkfft",
)


class GPUBaselineUnsupportedError(RuntimeError):
    """Raised by `make_fft_kernel(planner="gpu-*")` when the requested GPU
    baseline cannot represent `n` on this M2NDP target at all (see
    planning.gpu_baseline.common.BaselineStatus) -- carries the baseline's
    own full diagnostics, including the ORIGINAL GPU-chosen configuration
    it could not map, per that package's own "never silently substitute"
    rule. A caller comparing baselines wants this failure surfaced, not
    swallowed into a fallback plan."""


def _make_gpu_baseline_fft_kernel(
    n: int, *, planner: str, inverse: bool, batch: int, target: TargetProfile,
    output_path: str | Path | None, reference_check: bool, compute_lanes: int | None,
    narrow_middle_stages: bool, simd_lanes: int, spread_across_units: bool,
) -> Path:
    """`make_fft_kernel`'s own GPU-baseline path -- see that function's
    `planner` docstring. Builds the baseline plan via planning.
    gpu_baseline.{clfft,rocfft,vkfft}.plan(...), then renders it through
    the SAME codegen this project's own M2NDP-aware planner uses
    (codegen.fft_transpose_codegen.generate_recursive_fft_kernels) --
    planning still decides everything (this project's pre-existing
    invariant, unchanged); codegen only ever emits an already-fully-
    decided plan, GPU baseline or not."""
    from planning.gpu_baseline import clfft, rocfft, rocfft_default, vkfft
    from planning.gpu_baseline.common import BaselineStatus

    baseline_module = {
        "gpu-clfft": clfft,
        "gpu-rocfft": rocfft,
        "gpu-rocfft-tuned": rocfft,
        "gpu-rocfft-default": rocfft_default,
        "gpu-vkfft": vkfft,
    }[planner]
    result = baseline_module.plan(n, batch=batch, inverse=inverse, target=target)
    if result.status is not BaselineStatus.OK:
        raise GPUBaselineUnsupportedError(
            f"planner={planner!r} cannot represent n={n} (batch={batch}, "
            f"inverse={inverse}) on this M2NDP target:\n{result.diagnostics}"
        )

    if compute_lanes is None:
        compute_lanes = min(simd_lanes, target.lmul1_float32_lanes)

    source = generate_recursive_fft_kernels(
        result.plan, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        reference_check=reference_check, target=target, spread_across_units=spread_across_units,
    )

    if output_path is None:
        suffix = "_inverse" if inverse else ""
        planner_suffix = planner.replace("gpu-", "_")
        output_path = Path(__file__).resolve().parent / f"fft_fp32_N{n}{suffix}{planner_suffix}_generated.mojo"
    else:
        output_path = Path(output_path)

    output_path.write_text(source, encoding="utf-8")
    return output_path


def make_fft_kernel(
    n: int,
    *,
    inverse: bool = False,
    scratchpad_byte_budget: int | None = None,
    simd_lanes: int = 8,
    compute_lanes: int | None = None,
    narrow_middle_stages: bool = True,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
    cooperative_workers: int | str | None = None,
    batch: int = 1,
    output_path: str | Path | None = None,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    plan_index: int | None = None,
    reference_check: bool = True,
    verify_spill_free: bool = False,
    spill_probe_top_k: int = 1,
    rank_by_cycles: bool = False,
    spread_across_units: bool = True,
    mojo_root: str | None = None,
    m2ndp_root: str | None = None,
    planner: str = "current",
) -> Path:
    """Plan and render a length-`n` FFT kernel, write it to `output_path`
    (default: `fft_fp32_N{n}{_inverse}_generated.mojo` next to this file),
    and return the path written.

    `target`: the hardware/simulator constants this generator otherwise has
    no single source for (scratchpad capacity, LMUL=1 vector width, the
    interleave chunk cooperative leaves must divide evenly) -- see
    `TargetProfile`'s own docstring. Defaults to `DEFAULT_TARGET_PROFILE`,
    this repo's one real target today; pass a different profile only to
    plan against a different config.

    `plan_index`: `None` (the default) keeps today's exact single-heuristic
    plan (`make_recursive_transpose_plan` called directly, as if this
    parameter never existed). A given index instead builds candidate
    `plan_index` from `planning.search.fft_plan_search.generate_candidates` /
    `rank_candidates` (index 0 = lowest estimated cost) -- see that
    module's own docstring for what a candidate varies. When given,
    `cooperative_workers`/`tile_rows`/`tile_cols` are ignored: the chosen
    candidate's own choices already fully determine those (see `--dump-
    candidates` to see what each index actually is before picking one).

    `scratchpad_byte_budget`: how many bytes of one NDP unit's own
    scratchpad one leaf kernel may use -- the planner recurses (adds a
    tiled-transpose boundary) only once N would exceed this as a single
    fused kernel. `None` (the default) picks `4096` regardless of
    `cooperative_workers` -- tried making this cooperation-aware (a
    `16*sqrt(n)`-ish guess, see fft_plan_cooperative.
    default_cooperative_scratchpad_budget's own docstring) and measured it
    against the flat constant at three N: it won at two (N=4096, N=16384)
    but lost by ~26% at a third (N=1024, where keeping the *flat* 4096
    default outright beat every budget tried except one even bigger one
    that hit a register-spill wall) -- no formula found so far is safe to
    default to over just keeping `4096`, so this still does that. Pass a
    number explicitly to try something else for a specific N; see
    default_cooperative_scratchpad_budget's own docstring for the numbers
    behind this and what a real per-N search would need.
    `tile_rows`/`tile_cols`: physical transpose tile size, default
    `min(simd_lanes, ...)` picked by the planner itself (see
    make_recursive_transpose_plan) -- pass explicitly only to compare
    tile-size choices.

    `cooperative_workers`: `None` (the default) keeps every leaf exactly
    what this project produced before cooperative leaves existed (see
    CooperationPlan / fft_plan_cooperative.py). `"auto"` lets
    `choose_workers_per_fft` decide each leaf's own worker count from its
    own shape; a positive int caps that choice instead of overriding it
    outright (see `_build_leaf_kernel`'s own docstring in fft_plan_
    recursive.py for why it's a cap, not a raw override).

    `batch`: how many independent length-`n` transforms to run in one
    launch, `1` by default (today's exact prior behavior). The `batch`
    transforms sit back to back in one flat buffer -- see
    make_recursive_transpose_plan's own `batch` docstring for why every
    kernel this renders is already sized for it with no other planning
    change. Compatible with `--plan-index`: every candidate
    generate_candidates builds gets the same `batch`, since it's not
    itself a ranked search axis (it never changes which split/tier/
    worker/tile choice is legal or how they compare -- see that module's
    own note on this).

    `reference_check`: `True` (the default) keeps today's self-contained
    O(N^2) host DFT check baked into the generated kernel's own main() --
    fine for small correctness runs, but O(N^2) dwarfs the device kernels'
    own runtime long before N reaches the range plan comparison actually
    needs (confirmed directly: N=16384 with this on didn't finish inside a
    300s run timeout that ran to completion easily with it off). `False`
    instead dumps the random input and device output between INPUT_BEGIN/
    INPUT_END and OUTPUT_BEGIN/OUTPUT_END markers (see
    generate_recursive_fft_kernels's own docstring) for a Python-side
    numpy.fft check -- use this for any performance/cycle-comparison run
    at a large N; correctness at that N is established separately (the
    Python re-execution harness in verification/, or a real-toolchain run
    at a smaller N with reference_check left on).

    `simd_lanes` is the hardware launch granule (ties to `PooledRange`/
    `VECTOR_WIDTH` in src/m2ndp.mojo -- do not change it to tune register
    pressure, that miscounts how many microthreads get launched and
    silently produces an all-zero result). `compute_lanes` is the
    separate, safe knob for that: how wide a vector *instruction* each
    stage's arithmetic actually emits (see fft_codegen._chunk_batch).
    Defaults to `min(simd_lanes, target.lmul1_float32_lanes)` -- this target's
    LMUL=1 width -- since
    a full `simd_lanes=8` butterfly, fully unrolled the way this generator
    always renders one, needs more RVV registers (LMUL=2) than this
    target's simulator can spill through: it emits `csrr t, vlenb` to size
    the scalable-vector spill slot, an instruction the simulator's decoder
    does not implement, and the kernel panics the moment register pressure
    forces a spill. Pass `compute_lanes=simd_lanes` to opt back into the
    old, unchunked (and, on any kernel large enough to spill, simulator-
    incompatible) code shape.

    `narrow_middle_stages`: `True` (the default) halves `compute_lanes`
    (floor 1) for a leaf/near-FFT stage that is neither its own kernel's
    first nor last stage -- the shape a real N=630 register-pressure
    failure was isolated to (see generate_recursive_fft_kernels's own
    docstring).

    2026-08-28 update, important: this does *not* actually eliminate that
    spill -- direct probing found N=630/N=105's own radix-5 middle stage
    spills byte-for-byte identically (128-byte frame) whether this flag
    is `True` or `False`. It was briefly flipped to `False` by default
    that day on the theory that halving was neutral-to-harmful (it also
    turned out to *introduce* a spill for a plain radix-(4,4,4) leaf,
    e.g. N=64, that full width doesn't have) -- **and immediately
    reverted** after real-hardware testing showed `False` produces a
    silently WRONG answer for N=105/N=630 specifically, while `True`'s
    own byte-identical spill there has never once produced a wrong
    answer in any case tested. So: `True` is kept as the default not
    because it fixes anything, but because every real-hardware case
    tried so far (N=64, N=105, N=630) has been *correct* under it, and
    one (N=105/N=630) is confirmed *wrong* under `False`. `False` is
    kept as an override for exactly this reason -- comparing against it
    is what surfaces cases like N=64 where it's a strict improvement --
    but do not flip the default again without a real build+run
    correctness check across a broad N sweep, not just a spill-presence
    check (`spill_free` and "gives the right answer" are different
    questions -- see [[fft-spill-hard-filter]]).

    A *second*, unconditional liability -- radix 10/11/13/17's own operand
    count alone spills the same way regardless of stage position
    (N=11/13/17 confirmed MISMATCH standalone, no middle stage involved at
    all) -- is not controlled by this flag at all (see fft_codegen.
    _ALWAYS_NARROW_RADICES's own comment): this project has never confirmed
    those radices clean at compute_lanes=4 in any configuration, so
    `narrow_middle_stages=False` still narrows a stage using one of them.

    `verify_spill_free`: `False` (the default) keeps this function exactly
    what its own module docstring promises -- instant, toolchain-free,
    "just give it a number." `True` instead builds+runs the resolved plan
    against the real M2NDP-Detour toolchain (planning.diagnostics.spill_probe.
    probe_spill_free) before writing anything, closing the one gap every
    fix in narrow_middle_stages/_ALWAYS_NARROW_RADICES still leaves open:
    those eliminate every *known* spill-driven correctness liability, but
    a real spill can still occur elsewhere (confirmed 2026-08-27: the PRE/
    MIDDLE/POST transpose kernels' own tile-position address arithmetic --
    a scalar/GPR spill, not the `csrr ...,vlenb` vector liability those
    fixes target, and non-monotonic in tile size in a way this project
    doesn't yet have a static rule for -- see fft_plan_search.
    generate_tile_candidates' own tile sweep). Whether this searches or
    just checks depends on what else was requested:

    * `plan_index` given, or an explicit `tile_rows`/`tile_cols`/
      `cooperative_workers` override: the caller already picked one exact
      configuration, so this only *verifies* it -- raises planning.
      spill_probe.NoSpillFreeCandidateError (propagated, not swallowed) if
      that exact plan spills, rather than silently writing a kernel this
      project's own architecture (docs/STATUS.md: the whole register file
      is free, a spill should never be needed) says shouldn't exist.
    * Otherwise (the fully default heuristic path): searches
      `fft_plan_search.generate_candidates`'s own ranked candidates (split/
      radix-tier/tile/worker axes together) via planning.diagnostics.spill_probe.
      probe_and_rerank_candidates(top_k=spill_probe_top_k) for the
      cheapest *confirmed* spill-free one, and writes that plan instead of
      the bare heuristic pick if it differs. Still raises
      NoSpillFreeCandidateError if every candidate probed spills.

    Real cost: one or more full build+run round trips (tens of seconds to
    minutes each, same as `run_fft_test.sh`/`benchmark_fft_candidates.sh`)
    -- needs the real toolchain present (see scripts/env.sh), and is not
    something to enable for quick iteration. `spill_probe_top_k` (default
    1) caps how many candidates the search path probes before giving up;
    raise it to try harder before raising NoSpillFreeCandidateError, at
    the cost of more build+run rounds.

    `rank_by_cycles`: `False` (the default) picks, among the confirmed
    spill-free candidates the search path probed, the one `estimated_cost`
    ranked cheapest -- with `spill_probe_top_k=1` this is the *only* one
    probed, so it is whichever candidate the static cost model liked
    first that also happened to pass, not necessarily the fastest one on
    real hardware (estimated_cost is a cheap pre-filter, not a promise
    its order matches measured cycles -- see planning.diagnostics.spill_probe.
    probe_and_rerank_candidates' own `rank_by_cycles` docstring for the
    concrete N=16384 mismatch this is based on). `True` instead picks
    whichever of the probed candidates has the lowest real measured
    `ndp_cycles` -- only actually compares more than one candidate when
    `spill_probe_top_k > 1` (raise it together with this flag; at
    `spill_probe_top_k=1` there's nothing to compare, same pick either
    way). Only affects the search path (`explicit_choice` is always
    exactly one plan, nothing to rank).

    `spread_across_units`: `True` (the default since 2026-08-28) widens
    eligible non-cooperative launch rounds so they genuinely spread
    across more than one of the M2NDP config's own `num_ndp_units` (32)
    physical NDP units, instead of every launch landing on just 1 (this
    generator's own behavior before this flag existed). See
    codegen.fft_transpose_codegen.generate_recursive_fft_kernels's own
    `spread_across_units` docstring and `_safe_round_size`'s proof for
    why widening a round this way never gives one physical unit more
    microthreads than its own scratchpad (`max_uthread`) allows.
    Confirmed real-hardware: a ~2.6x wall-clock speedup at N=256/batch=64
    (238.95s -> 92.01s, 4 separate simulator sessions collapsed to 1 --
    `ndp_cycles` alone reports these as ~equal, since it only reflects
    the *last* session's own internal clock, not the real cost of
    spinning up several; see the multi-NDP-unit-parallelism plan's own
    N3 section), and correct output at 3 scales (small/multi-unit,
    exactly at the interleave-wraparound boundary, well past it -- see
    that plan's own N4 section) via real build+run+reference-check, not
    just "didn't crash." Pass `False` (`--no-spread-across-units` on the
    CLI) to compare against the old, pre-2026-08-28 single-unit shape.
    Never affects cooperative-worker leaves (untouched, out of scope --
    see that plan's own C-track, not yet investigated).

    `mojo_root`/`m2ndp_root`: passed straight through to spill_probe --
    `None` picks the same defaults `scripts/env.sh` does (see that
    function's own docstring); override only to probe against a different
    toolchain build (e.g. a `git worktree`).

    `planner`: `"current"` (the default) is this exact function's own
    prior behavior, unchanged -- the M2NDP-aware recursive planner
    (optionally its own candidate search / spill-verified path, see
    `plan_index`/`verify_spill_free` above). One of `GPU_BASELINE_PLANNERS`
    (`"gpu-clfft"`, `"gpu-rocfft"`, `"gpu-vkfft"`) instead builds the
    corresponding GPU-derived BASELINE plan (planning.gpu_baseline.*) --
    see that package's own module docstrings for what each one ports and
    from where. A GPU baseline planner IGNORES every M2NDP-search-specific
    parameter above (`scratchpad_byte_budget`, `plan_index`,
    `cooperative_workers`, `tile_rows`/`tile_cols`, `verify_spill_free`,
    `spill_probe_top_k`, `rank_by_cycles`) -- those are M2NDP-aware tuning
    knobs this baseline must stay independent of (see gpu_baseline/
    common.py's own non-negotiable-rule docstring); passing any of them
    together with a GPU `planner` raises rather than silently ignoring
    the conflict. A GPU baseline that cannot represent `n` on this target
    (see planning.gpu_baseline.common.BaselineStatus) raises a
    `GPUBaselineUnsupportedError` carrying the full diagnostic, rather
    than silently falling back to `"current"`.
    """
    if planner in GPU_BASELINE_PLANNERS:
        conflicting = {
            "scratchpad_byte_budget": scratchpad_byte_budget,
            "plan_index": plan_index,
            "cooperative_workers": cooperative_workers,
            "verify_spill_free": verify_spill_free or None,
        }
        set_conflicts = {k: v for k, v in conflicting.items() if v not in (None, False)}
        if set_conflicts:
            raise ValueError(
                f"planner={planner!r} is a GPU baseline and must stay independent of "
                f"M2NDP-aware tuning knobs, but these were also given: {set_conflicts} "
                f"-- see make_fft_kernel's own `planner` docstring"
            )
        return _make_gpu_baseline_fft_kernel(
            n, planner=planner, inverse=inverse, batch=batch, target=target,
            output_path=output_path, reference_check=reference_check,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            simd_lanes=simd_lanes, spread_across_units=spread_across_units,
        )

    if n < 2:
        # A length-1 "FFT" needs zero radix stages, which _build_plan/
        # _check_layouts (planning/fft_plan_core.py) has never supported --
        # not something introduced here, and not worth adding a zero-stage
        # passthrough kernel for. Fail clearly instead of surfacing that
        # internal ValueError.
        raise ValueError(f"n={n} is too small: make_fft_kernel needs n >= 2")
    if compute_lanes is None:
        compute_lanes = min(simd_lanes, target.lmul1_float32_lanes)
    if scratchpad_byte_budget is None:
        scratchpad_byte_budget = 4096
    explicit_choice = (
        plan_index is not None or tile_rows is not None or tile_cols is not None
        or cooperative_workers is not None
    )
    if plan_index is None:
        plan = make_recursive_transpose_plan(
            n,
            scratchpad_byte_budget=scratchpad_byte_budget,
            simd_lanes=simd_lanes,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            inverse=inverse,
            batch=batch,
            spad_capacity_bytes=target.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
            cooperative_workers=cooperative_workers,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
        )
    else:
        plan = plan_for_index(
            n, plan_index, inverse=inverse, target=target, batch=batch,
            scratchpad_byte_budget=scratchpad_byte_budget, simd_lanes=simd_lanes,
        )

    if verify_spill_free:
        # Local import: this is the one call site in the whole module that
        # needs the real toolchain, and every other caller of this file
        # (verification/, the CLI's own --dump-plan/--dump-candidates
        # paths) must keep working with no toolchain present at all -- see
        # planning/spill_probe.py's own module docstring for the same
        # "toolchain-free unless a caller opts in" discipline.
        from planning.diagnostics.spill_probe import NoSpillFreeCandidateError, probe_and_rerank_candidates, probe_spill_free

        if explicit_choice:
            result = probe_spill_free(
                plan, compute_lanes=compute_lanes, simd_lanes=simd_lanes,
                narrow_middle_stages=narrow_middle_stages, target=target,
                mojo_root=mojo_root, m2ndp_root=m2ndp_root,
            )
            if not (result.build_ok and result.run_ok):
                raise RuntimeError(
                    f"verify_spill_free: the toolchain probe itself failed for n={n} "
                    f"(build_ok={result.build_ok} run_ok={result.run_ok}) -- see its own log:\n{result.log}"
                )
            if not result.spill_free:
                tile_suggestions = _no_tail_tile_suggestions(plan)
                suggestion_block = (
                    f"\n\nStatically safe tile alternatives for this plan's own "
                    f"transpose shape(s) (no real probe needed to know these have no "
                    f"tail branch, though --verify-spill-free would still confirm this "
                    f"exact combination):\n{tile_suggestions}"
                    if tile_suggestions else ""
                )
                raise NoSpillFreeCandidateError(
                    f"verify_spill_free: the exact plan requested for n={n} "
                    f"(plan_index={plan_index} tile_rows={tile_rows} tile_cols={tile_cols} "
                    f"cooperative_workers={cooperative_workers}) spills "
                    f"({result.spilling_kernels}) -- refusing to write a kernel this "
                    f"project's own architecture (docs/STATUS.md) says shouldn't spill at all."
                    f"{suggestion_block}\n\n"
                    f"Or drop the explicit plan_index/tile_rows/tile_cols/cooperative_workers "
                    f"override entirely and keep --verify-spill-free alone: that path searches "
                    f"generate_candidates' own ranked candidates and returns the first one "
                    f"actually confirmed spill-free (raise --spill-probe-top-k to try harder, "
                    f"and add --rank-by-cycles to prefer the fastest *measured* one among "
                    f"however many it probes, not just the cheapest by estimated_cost) -- or "
                    f"drop verify_spill_free entirely to accept this exact plan as-is anyway.",
                    kept=[], excluded=[result], unresolved=[],
                )
        else:
            ranked = rank_candidates(generate_candidates(
                n, target=target, inverse=inverse, batch=batch,
                scratchpad_byte_budget=scratchpad_byte_budget, simd_lanes=simd_lanes,
            ))
            probed = probe_and_rerank_candidates(
                ranked, compute_lanes=compute_lanes, simd_lanes=simd_lanes,
                narrow_middle_stages=narrow_middle_stages, target=target,
                top_k=spill_probe_top_k, rank_by_cycles=rank_by_cycles,
                mojo_root=mojo_root, m2ndp_root=m2ndp_root,
            )
            plan = probed.candidates[0].plan

    source = generate_recursive_fft_kernels(
        plan, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        reference_check=reference_check, target=target, spread_across_units=spread_across_units,
    )

    if output_path is None:
        suffix = "_inverse" if inverse else ""
        output_path = Path(__file__).resolve().parent / f"fft_fp32_N{n}{suffix}_generated.mojo"
    else:
        output_path = Path(output_path)

    output_path.write_text(source, encoding="utf-8")
    return output_path


def plan_for_index(
    n: int, plan_index: int, *, inverse: bool, target: TargetProfile,
    scratchpad_byte_budget: int, simd_lanes: int, batch: int = 1,
) -> RecursiveFFTPlan:
    """The `plan_index`'th plan from `generate_candidates`/`rank_candidates`
    (index 0 = lowest estimated cost) -- shared by `make_fft_kernel` and
    the `--dump-plan --plan-index` CLI path so both resolve exactly the
    same candidate the same way."""
    ranked = rank_candidates(generate_candidates(
        n, target=target, inverse=inverse, batch=batch,
        scratchpad_byte_budget=scratchpad_byte_budget, simd_lanes=simd_lanes,
    ))
    if not (0 <= plan_index < len(ranked)):
        raise ValueError(
            f"plan_index={plan_index} out of range: generate_candidates(n={n}) "
            f"produced {len(ranked)} candidate(s) (0..{len(ranked) - 1})"
        )
    return ranked[plan_index].plan


def _ad_hoc_candidate(n: int, inverse: bool, plan: RecursiveFFTPlan) -> FFTPlanCandidate:
    """Wrap an already-built plan (e.g. `--dump-plan` without `--plan-index`,
    built via explicit `--cooperative-workers`/`--tile-rows`/`--tile-cols`
    overrides `generate_candidates`'s own sweep never produces) for
    `format_plan_summary` -- metrics only, no ranked-candidate index since
    this plan was never part of a `generate_candidates` call."""
    metrics = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
    metrics = dataclasses.replace(metrics, estimated_cost=estimate_cost(metrics))
    return FFTPlanCandidate(
        n=n, inverse=inverse, plan=plan,
        choices=PlanChoices(split_near_length=None, radix_tier_name="n/a (explicit overrides)",
                             workers_per_fft=None, tile=None),
        metrics=metrics,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a working M2NDP FFT Mojo kernel for length N."
    )
    parser.add_argument("n", type=int, help="FFT length (any product of supported radices)")
    parser.add_argument(
        "--planner", type=str, default="current",
        choices=("current", *GPU_BASELINE_PLANNERS),
        help='"current" (default): this project\'s own M2NDP-aware recursive planner. '
        '"gpu-clfft"/"gpu-rocfft"/"gpu-vkfft": build a GPU-derived BASELINE plan instead '
        "(planning.gpu_baseline.*), ignoring every M2NDP-search-specific flag below -- "
        "see make_fft_kernel's own `planner` docstring. Raises GPUBaselineUnsupportedError "
        "if the requested baseline cannot represent this N on this target at all.",
    )
    parser.add_argument("--inverse", action="store_true", help="generate the inverse FFT")
    parser.add_argument(
        "--scratchpad-byte-budget", type=int, default=None,
        help="bytes of one NDP unit's own scratchpad one leaf kernel may use "
        "(default: 4096, same whether or not --cooperative-workers is given -- "
        "see make_fft_kernel's own docstring on why a cooperation-aware "
        "default isn't safe yet)",
    )
    parser.add_argument(
        "--cooperative-workers", type=str, default=None,
        help='"auto" to let each leaf pick its own cooperative worker count '
        "(see choose_workers_per_fft), or an integer to cap it -- default: "
        "off, one uthread per sub-FFT (this project's original behavior)",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="how many independent length-N transforms to run in one launch, "
        "back to back in one flat buffer -- default: 1 (a single transform, "
        "today's prior behavior)",
    )
    parser.add_argument("--simd-lanes", type=int, default=8, help="SIMD lane count (default: 8)")
    parser.add_argument(
        "--compute-lanes", type=int, default=None,
        help="SIMD width the emitted arithmetic actually uses, independent of "
        "--simd-lanes (the launch granule -- do not use this to change that). "
        f"Default: min(simd_lanes, {DEFAULT_TARGET_PROFILE.lmul1_float32_lanes}), this target's LMUL=1 "
        "width, to avoid register-spill instructions the simulator can't run. "
        "Pass --compute-lanes matching --simd-lanes to opt back into the old, "
        "unchunked code shape.",
    )
    parser.add_argument(
        "--no-narrow-middle-stages", action="store_true",
        help="disable halving --compute-lanes (floor 1) for a stage that is "
        "neither its own kernel's first nor last -- default: on, since this "
        "is the shape a real N=630 register-pressure failure was isolated "
        "to (see make_fft_kernel's own narrow_middle_stages docstring); pass "
        "this to compare against the old flat-compute_lanes code shape. Does "
        "NOT affect radix 10/11/13/17, which are always narrowed regardless "
        "of stage position -- see the same docstring",
    )
    parser.add_argument("--tile-rows", type=int, default=None, help="transpose tile rows (default: planner's own choice)")
    parser.add_argument("--tile-cols", type=int, default=None, help="transpose tile cols (default: planner's own choice)")
    parser.add_argument(
        "--verify-spill-free", action="store_true",
        help="build+run the resolved plan against the real M2NDP-Detour toolchain "
        "before writing anything, and refuse to write a kernel that spills -- see "
        "make_fft_kernel's own verify_spill_free docstring. Slow (a real build+run "
        "round trip, or several if searching); needs the toolchain present. Default: off.",
    )
    parser.add_argument(
        "--spill-probe-top-k", type=int, default=1,
        help="with --verify-spill-free and no --plan-index/--tile-rows/--tile-cols/"
        "--cooperative-workers override: how many confirmed spill-free candidates to "
        "search for before giving up (default: 1 -- just the cheapest one)",
    )
    parser.add_argument(
        "--rank-by-cycles", action="store_true",
        help="with --verify-spill-free's search path: pick the probed candidate with "
        "the lowest real measured ndp_cycles instead of the cheapest by estimated_cost "
        "-- see make_fft_kernel's own rank_by_cycles docstring. Only compares more than "
        "one candidate when --spill-probe-top-k > 1; raise that together with this. "
        "Default: off (cheapest-by-estimated-cost, today's behavior).",
    )
    parser.add_argument(
        "--no-spread-across-units", action="store_true",
        help="disable widening eligible non-cooperative launch rounds across more than one "
        "physical NDP unit -- default: on since 2026-08-28 (confirmed ~2.6x real wall-clock "
        "speedup, see make_fft_kernel's own spread_across_units docstring); pass this to "
        "compare against the old, single-unit-only shape. Never affects cooperative-worker "
        "leaves either way.",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="output .mojo path")
    parser.add_argument(
        "--dump-candidates", action="store_true",
        help="print every candidate plan planning.search.fft_plan_search.generate_candidates "
        "finds for N (ranked by estimated cost, see format_plan_summary) and exit "
        "without writing a .mojo file",
    )
    parser.add_argument(
        "--dump-plan", action="store_true",
        help="print the plan that would be built (today's default heuristic, or "
        "--plan-index's pick) and exit without writing a .mojo file",
    )
    parser.add_argument(
        "--plan-index", type=int, default=None,
        help="build candidate K from generate_candidates's ranked list (0 = lowest "
        "estimated cost) instead of the default single-heuristic plan -- see "
        "--dump-candidates to see what each index is first. Ignores "
        "--cooperative-workers/--tile-rows/--tile-cols: the chosen candidate's own "
        "choices already determine those",
    )
    parser.add_argument(
        "--no-reference-check", action="store_true",
        help="skip the self-contained O(N^2) host DFT check and instead dump "
        "INPUT/OUTPUT markers for a Python-side numpy.fft check -- use for a "
        "large-N performance/cycle-comparison run (see make_fft_kernel's own "
        "reference_check docstring; O(N^2) dominates wall-clock well before N "
        "reaches the range plan comparison needs)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    cooperative_workers: int | str | None = args.cooperative_workers
    if cooperative_workers is not None and cooperative_workers != "auto":
        cooperative_workers = int(cooperative_workers)
    scratchpad_byte_budget = args.scratchpad_byte_budget if args.scratchpad_byte_budget is not None else 4096

    if args.dump_candidates:
        ranked = rank_candidates(generate_candidates(
            args.n, inverse=args.inverse, batch=args.batch,
            scratchpad_byte_budget=scratchpad_byte_budget, simd_lanes=args.simd_lanes,
        ))
        for i, candidate in enumerate(ranked):
            print(format_plan_summary(candidate, index=i))
            print()
        return

    if args.dump_plan:
        if args.plan_index is not None:
            ranked = rank_candidates(generate_candidates(
                args.n, inverse=args.inverse, batch=args.batch,
                scratchpad_byte_budget=scratchpad_byte_budget, simd_lanes=args.simd_lanes,
            ))
            if not (0 <= args.plan_index < len(ranked)):
                raise ValueError(
                    f"--plan-index={args.plan_index} out of range: "
                    f"generate_candidates produced {len(ranked)} candidate(s)"
                )
            print(format_plan_summary(ranked[args.plan_index], index=args.plan_index))
        else:
            plan = make_recursive_transpose_plan(
                args.n, scratchpad_byte_budget=scratchpad_byte_budget, simd_lanes=args.simd_lanes,
                tile_rows=args.tile_rows, tile_cols=args.tile_cols, inverse=args.inverse,
                batch=args.batch,
                spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
                max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
                cooperative_workers=cooperative_workers,
                interleave_chunk_uthreads=DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
            )
            print(format_plan_summary(_ad_hoc_candidate(args.n, args.inverse, plan)))
        return

    path = make_fft_kernel(
        args.n,
        inverse=args.inverse,
        scratchpad_byte_budget=args.scratchpad_byte_budget,
        simd_lanes=args.simd_lanes,
        compute_lanes=args.compute_lanes,
        narrow_middle_stages=not args.no_narrow_middle_stages,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        cooperative_workers=cooperative_workers,
        batch=args.batch,
        output_path=args.output,
        plan_index=args.plan_index,
        reference_check=not args.no_reference_check,
        verify_spill_free=args.verify_spill_free,
        spill_probe_top_k=args.spill_probe_top_k,
        rank_by_cycles=args.rank_by_cycles,
        spread_across_units=not args.no_spread_across_units,
        planner=args.planner,
    )
    print(f"generated: {path}")


if __name__ == "__main__":
    main()
