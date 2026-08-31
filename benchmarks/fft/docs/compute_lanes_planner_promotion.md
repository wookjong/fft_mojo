# compute_lanes promoted to a planning decision (Phase 5, 2026-08-31)

## What changed

`compute_lanes` (how wide a vector *instruction* one stage's arithmetic
renders as -- see `make_fft_kernel.py`'s own discussion) used to be
decided live, at codegen emit time, by `codegen.fft_codegen.
_stage_compute_lanes`: given a flat `compute_lanes`/`narrow_middle_stages`
plus one stage's own radix/position, it computed that stage's own render
width on the spot. That is a *planning* decision made inside codegen,
which this project's own architecture wants codegen never to do.

`FFTStagePlan` gained a `compute_lanes: int | None = None` field.
`planning/fft_plan_lanes.py` is the new single owner of the heuristic
(moved verbatim from `fft_codegen.py`, same three reasons, same real-
hardware evidence, same behavior):

- `resolve_stage_compute_lanes(...)` -- the exact per-stage logic,
  relocated.
- `apply_compute_lanes(plan, *, compute_lanes, narrow_middle_stages)` --
  returns a plan with every stage's own `compute_lanes` field populated.
- `generate_compute_lane_candidates(plan, *, compute_lanes)` -- a small,
  bounded set (baseline / unnarrowed / all-scalar, deduped), not a
  `{1,2,4}^stage_count` explosion (the plan's own explicit "lane
  candidate explosion 방지" rule -- narrower is not always safer, e.g.
  persistent leaves preferring wide lanes).

Codegen now prefers `stage.compute_lanes` when the planner already set
it, in every place that used to call `_stage_compute_lanes` live:
`fft_codegen._emit_stage` and `fft_persistent_codegen.emit_stage_phase`.
`_stage_compute_lanes` itself is now a thin backward-compatible wrapper
around `fft_plan_lanes.resolve_stage_compute_lanes` -- every existing
caller that never populates `stage.compute_lanes` (i.e. every caller that
existed before this phase) gets byte-for-byte identical output, confirmed
directly (see Verification below), not merely "should still work."

`fft_plan_search._plan_signature` now folds each leaf's own per-stage
`compute_lanes` tuple into its signature, so two candidates that differ
only in lane width no longer collapse into "the same plan" during dedup.

## Why this is additive, not a default-behavior change

`make_fft_kernel.py`'s own default pipeline is untouched: it still passes
flat `compute_lanes`/`narrow_middle_stages` straight to codegen, every
stage's own `compute_lanes` field stays `None`, and codegen falls back to
the live computation exactly as before. Wiring the planner-driven path
into that default pipeline's own candidate search (so `fft_plan_search.py`
actually generates/ranks/selects per-stage lane candidates) is Phase 6
("radix x execution strategy x compute_lanes joint search"), not this
phase -- this phase makes the mechanism exist and proves it's correct.

## Verification

- **Equivalence**: for a representative case (N=105, radices (3,5,7),
  known middle-stage-narrowing shape), `apply_compute_lanes` followed by
  codegen using `stage.compute_lanes` produces **byte-for-byte identical**
  generated text to the old live-resolution path (`compute_lanes=4,
  narrow_middle_stages=True` passed straight to codegen) -- confirmed by
  direct diff, both for a single fused leaf and for both leaves of a real
  N=960 recursive split.
- **Candidates differ**: `generate_compute_lane_candidates` on that same
  case produces 3 distinct per-stage lane tuples (`[4,2,4]`, `[4,4,4]`,
  `[1,1,1]`) and 3 distinct rendered kernel texts.
- **Real hardware**: a plan built via `apply_compute_lanes` (planner-
  resolved lanes, codegen making no width decision) builds and runs
  correctly on the real M2NDP-Detour simulator (N=105, `[host] FFT
  verification passed`).
- Full existing regression suite (`verify_fft_plan.py`) unaffected --
  every existing caller still resolves lanes the old way and gets
  identical output.

No `third_party/m2ndp-detour` source touched.
