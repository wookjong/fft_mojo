# Tests

`scripts/test.sh` is the entry point. It runs a hierarchy of tiers, cheapest
first, stopping at the first failure. `tests/manifest.tsv` lists the cases.

```
./scripts/test.sh            # every tier, then the opcode-coverage report
./scripts/test.sh t0 t1      # only these
./scripts/test.sh coverage   # just the opcode-coverage map
REQUIRE_ALL_GREEN=1 ./scripts/test.sh   # require every case, xfail included
```

## Tiers

| Tier | What it covers | How it checks |
|------|----------------|---------------|
| t0 | `llc` / `ld.lld`, the riscv64 target and the xm2ndp extension | presence and target probes |
| t1 | device IR + assembly for every workload | static checks (`build.sh` then `verify.sh`) |
| t2 | device_main paths on Detour: compute-only, a function call, a launch | golden + simulator stats |
| t3 | one-operation kernels (fexp, indexed atomic, reduce, ...) | golden + the unit that should have run |
| t4 | whole workloads on Detour's controller | golden + memory traffic |

## Two things a run checks

- **golden** -- the output bytes equal the expected ones.
- **stat** -- the simulator's counters show the work actually happened: cycles
  scale with the input, the load/store traffic is there, an exp lands on the SF
  unit, a device_main-only case launches nothing while a launch case launches
  what it should. A right answer with no traffic is a modeling bug the golden
  alone does not catch.

## Manifest

Tab-separated, one row per case: `name`, the `tiers` it belongs to, its
`family` (`C` controller/device_main, `O` operation, `W` workload), the `check`
a run asserts (`golden`/`stat`), the `expect` (`pass` or `xfail`), and a `note`
saying why for an xfail. Adding a case is a new row.

The full 24-workload set is the coverage target — the breadth the Spike executor
port validated. `expect` is the single place a case's status lives, so the suite
is honest today and flips to all-green as fixes land: mark a fixed case `pass`.
A case marked `xfail` that starts passing is reported `[XPASS]`, a reminder to
promote it.

`REQUIRE_ALL_GREEN=1` is the one switch that ignores `expect` and requires every
case to pass — the gate to turn on the day the in-progress simulator fixes land.

t2 and t3 rows wait on a Detour harness that returns the stat counters
(`NdpStats`: per-unit issue counts, scratchpad reads/writes, launch count)
rather than only printing them; until it lands those tiers report pending.

## Opcode coverage

`./scripts/test.sh coverage` (or `./scripts/opcode-map.sh`) reports which
RVV / scalar / CSR / M2NDP opcodes each workload lowers to, read from the device
`.s` our llc emits — the same assembly a launch feeds the loader. It also cross-
references the featured opcode families the Spike port validated (vec↔scalar
moves, broadcasts, int↔fp converts, slides, the FMA family, `frm` via `csrrwi`,
the M2NDP vector and scalar-fp atomics, reductions, gather) against the workloads
that exercise each. `docs/TESTING.md` carries the rendered map.

## Where the boundary is

M2NDP-Detour carries its own assembly-level tests -- the decoder and loader over
committed object fixtures, run through `ctest`. This tree owns the pipeline above
that: compiling a workload, checking the code it generates, and running it on
Detour's controller with the simulator watching the traffic.
