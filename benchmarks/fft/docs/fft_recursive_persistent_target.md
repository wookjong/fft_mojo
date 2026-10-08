# Recursive and mixed persistent target propagation

Execution-strategy implementation step 2 (2026-10-07).

Previously `_build_leaf_kernel` omitted `target` when calling
`make_persistent_leaf_plan`. Recursive persistent leaves and mixed worker
sequences therefore used the default 32-unit profile even when native search
received a 4-unit profile. Explicit recursive scratchpad limits were also
ignored by persistent leaves.

`make_recursive_transpose_plan(..., target=profile)` now resolves the profile
once and passes it through every recursive level and persistent leaf. Unit
count, interleave width, address mapping, scratchpad capacity and registration
limit reach the same factory used by standalone persistent plans. No second
implementation of persistent feasibility is introduced.

Explicit `spad_capacity_bytes`, `max_concurrent_scratchpad_bytes` and
`interleave_chunk_uthreads` override the supplied profile; omitted values inherit
it. If `target` is omitted, plain/cooperative leaves retain their legacy uncapped
resource defaults, while persistent leaves use the default profile with any
explicit resource overrides. An interleave override must remain consistent with
the profile's mapping stride; the existing persistent factory rejects an
inconsistent combination. The concurrent-scratchpad field retains the existing
per-uthread throttle semantics; this change does not introduce a new persistent
throttle.

Native search, direct kernel generation and benchmark planning forward their
profiles. Per-leaf search skips candidates rejected by the authoritative factory
instead of aborting the search. Registration validation from step 1 remains in
place before ranking.

Verification: `python -m verification.verify_fft_recursive_target` covers
standalone equivalence, deep recursion, mixed forward/inverse numerical execution,
legacy resource overrides, small scratchpad and registration limits, actual
search candidates, custom coherent interleaving and default compatibility.
Numerical checks interpret emitted kernel bodies and compare with NumPy; no
Mojo binary or M2NDP simulator execution was available in this environment.
