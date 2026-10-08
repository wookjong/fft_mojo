# FFT launch granule support (execution step 5)

`simd_lanes` is the Float32 pool launch granule. `compute_lanes` controls
arithmetic width within emitted FFT stages. Changing the former to 4 is not
the supported way to request 4-lane arithmetic in the default unit-spreading
execution path.

The existing spreading calculation assumes one 32-byte microthread, or
`simd_lanes=8`. It previously failed late in recursive host rendering when
planning or CLI dump modes had already accepted the incompatible combination.

The common `validate_fft_launch_granule` now checks positive integer granules
and rejects `spread_across_units=True` with a granule other than 8. Recursive
planning and candidate search expose the intended `spread_across_units` policy
for early validation; their default False preserves existing non-spreading
experiments. These planner arguments validate only and do not store a launch
policy: pass the same policy to codegen. The high-level generator and CLI pass
it consistently, including candidate selection and dump paths. High-level
execution continues to default to spreading enabled.

Recursive codegen checks stage round eligibility before emitting device bodies.
Existing cooperative and persistent stages retain their exemptions from round
widening; a manually assembled tree containing only exempt stages preserves
its existing behavior. High-level request validation conservatively requires
8 when unit spreading is requested, regardless of the selected strategy.

Examples:

```sh
# Supported default launch granule with four arithmetic lanes.
python make_fft_kernel.py 64 --simd-lanes 8 --compute-lanes 4

# Existing non-spreading launch-granule experiment.
python make_fft_kernel.py 64 --simd-lanes 4 --no-spread-across-units
```

The latter remains an experimental low-level option, not a claim of simulator
support for a different hardware launch granule. Error messages explain both
options. CLI invalid combinations exit with an argparse error before planning,
candidate dumping or file writes.

`python -m verification.verify_fft_launch_support` contains seven checks for
planner/search/API/CLI/codegen rejection, early failure before costing or
emission, no writes on invalid requests, valid source generation, positive
integer checks and forward/inverse numerical equivalence for both supported
rendering paths. Output writes are intercepted in these tests. Numerical checks
interpret emitted bodies and compare against NumPy; actual Mojo/simulator
execution was unavailable.
