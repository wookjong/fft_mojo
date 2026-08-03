# Tests

`scripts/test.sh` is the entry point. It runs a hierarchy of tiers, cheapest
first, stopping at the first failure. `tests/manifest.tsv` lists the cases.

```
./scripts/test.sh            # every tier
./scripts/test.sh t0 t1      # only these
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

Tab-separated: `name`, the tiers a case belongs to, its family (`C`
controller/device_main, `O` operation, `W` workload), and what a run asserts.
Adding a case is a new row.

t2 and t3 rows wait on a Detour harness that returns the stat counters
(`NdpStats`: per-unit issue counts, scratchpad reads/writes, launch count)
rather than only printing them; until it lands those tiers report pending.

## Where the boundary is

M2NDP-Detour carries its own assembly-level tests -- the decoder and loader over
committed object fixtures, run through `ctest`. This tree owns the pipeline above
that: compiling a workload, checking the code it generates, and running it on
Detour's controller with the simulator watching the traffic.
