# Cooperative worker validation (execution step 3)

## Support contract

A cooperative FFT shares scratchpad on one physical unit. Its worker count
must be a positive integer divisor of the target interleave chunk. For the
default 8-thread chunk, legal counts are 1, 2, 4 and 8. Counts such as 3 and
16 can split the logical FFT across physical units and are rejected. This rule
applies to cooperative execution; persistent logical workers retain their
existing temporal-wave semantics.

Factory legality is distinct from search usefulness: a legal worker count can
leave workers idle on small stages. Candidate search still applies its existing
busiest-stage bound. Recursive `cooperative_workers` remains an upper-bound
heuristic; explicit factory and tail worker counts are validated without rounding.

## Implementation

`validate_cooperative_workers` is the common grouping check. The direct leaf
factory and explicit cooperative hybrid-tail selector use it. Explicit tail
counts are checked before the no-tail fast path, so selector validity does not
change with batch size. Irrelevant tail-worker arguments on other tail strategies
retain their previous behavior.

The leaf factory and partition candidate generator accept
`interleave_chunk_uthreads` (default 8). Recursive builders and GPU cooperative
mapping pass the actual target chunk. Hybrid-tail automatic selection and leaf
construction also use the target chunk. Invalid chunk values produce ValueError.

## Verification

`python -m verification.verify_fft_cooperative_workers` contains five regression
tests covering invalid direct and tail counts, no-tail boundaries, custom chunks,
invalid chunk types, and emitted-body numerical execution for W=1/2/4/8 with
all three partition modes in both transform directions. Existing hybrid-tail,
recursive-target, search, execution-invariant and GPU golden suites also pass.
No Mojo binary or simulator run was available; numeric validation interprets
emitted kernel bodies and compares their results against NumPy.
