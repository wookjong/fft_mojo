# Testing

`scripts/test.sh` is the one entry point. It runs a tier hierarchy cheapest-first
and stops at the first hard failure, printing one PASS/FAIL line per case. Cases
come from `tests/manifest.tsv`. `tests/README.md` covers the tiers in depth; this
note is the map of what the suite covers and how to read it.

## Running it

```
./scripts/test.sh                        # every tier, then the coverage report
./scripts/test.sh t0 t1                  # only these
./scripts/test.sh coverage               # just the opcode-coverage map
REQUIRE_ALL_GREEN=1 ./scripts/test.sh    # require every case, xfail included
```

The suite builds no toolchain: `MOJO_ROOT` points at a Mojo install, `build/`
links our LLVM (`llc`/`ld.lld`), and `M2NDP_DET` points at the M2NDP-Detour tree
whose `build/bin/m2ndp_run` runs a task. CI (`.github/workflows/test.yml`) runs
`./scripts/test.sh` inside the image that carries all three.

## Tiers

| Tier | Covers | Checks |
|------|--------|--------|
| t0 | `llc` / `ld.lld`, the riscv64 target, the xm2ndp extension | presence + target probes |
| t1 | device IR + assembly for every workload | `build.sh` then `verify.sh` static checks |
| t2 | device_main on Detour's controller | launches=0, load/store traffic, no abort |
| t3 | one-operation kernels (fexp, atomic, reduce) | pending the stat harness |
| t4 | whole workloads end-to-end on the controller | golden self-check via `host-run.sh` |
| coverage | the opcodes each workload lowers to | a report, not a gate |

## Coverage target and the expected-status manifest

The coverage target is the full 24-workload set in `benchmarks/` — the same
breadth the Spike executor port validated. Every workload is a t4 row in
`tests/manifest.tsv`; `devmain_self` is the t2 controller fixture.

Each row carries an `expect` column: `pass` (must pass) or `xfail` (known-broken
today, run anyway so the failure stays visible). `expect` is the single place a
case's status lives. Ten workloads pass on the base executor today; the rest are
`xfail` pending in-progress simulator fixes for the RVV opcodes the base executor
mishandles (the `RENAMING PANIC`), and `spmv`, whose exact float check rounds
~1e-5 off. As each fix lands, the case's row flips to `pass`; a case still marked
`xfail` that starts passing is reported `[XPASS]` so it never stays stale.

`REQUIRE_ALL_GREEN=1` is the one switch that ignores `expect` and requires every
case to pass — the gate to turn on the day the fixes land.

## Opcode-coverage map

`./scripts/opcode-map.sh` reads the device `.s` our llc emits — the same
assembly a launch feeds the loader — and reports the distinct opcodes each
workload lowers to. It is the evidence the suite covers the opcode surface the
port validated. The map is generated, not hand-kept; the snapshot below is what
`./scripts/opcode-map.sh --md` prints today.

### Featured families the Spike port validated

| Opcode family (port-validated) | Workloads exercising it |
|---|---|
| vec→scalar move (vfmv.f.s/vmv.x.s) | imdb_gt_lt_fp32, imdb_gteq_lt_int64, imdb_lt_int64, kmeans_assign, layernorm, softmax |
| scalar→vec broadcast (vfmv.v.f/vmv.v.x) | gelu, imdb_gteq_lt_int64, layernorm, memset, softmax, vector_exp |
| int↔fp convert (vfcvt.x.f.v/f.x.v) | softmax, vector_exp |
| fp narrow/widen (vfncvt/vfwcvt) | narrow, wide |
| slide (vslidedown/vslideup) | imdb_gteq_lt_int64, imdb_lt_int64, layernorm, softmax |
| FMA family (vfmacc/vfmadd/vfmsac) | gelu, layernorm, softmax, vector_exp |
| reduction (vfredmax/vredmin/...) | kmeans_assign, softmax |
| CSR frm (csrrwi frm, fsrm/fsrmi) | softmax, vector_exp |
| M2NDP vector AMO (int/fp) | gemv, gemv_aggregation, histogram |
| M2NDP scalar-fp AMO (famoadd/famomax) | layernorm, softmax |
| M2NDP kernel exit | every workload |
| mask/predicate (vmslt/vmfgt/vmand.mm/...) | imdb_gt_lt_fp32, imdb_gteq_lt_int64, imdb_lt_int64, kmeans_assign, relu, softmax, vector_exp |
| gather (vrgather) | kmeans_assign |
| vector fp arithmetic (vfadd/vfmul/...) | dlrm_sls, gelu, gemv, layernorm, residual, softmax, vector_exp, wide |
| vector int arithmetic (vadd/vand/vsll/...) | gemv, gemv_aggregation, histogram, imdb_three_col_and, imdb_two_col_and, softmax, vector_add, vector_exp |
| vector load/store (vle/vse) | dlrm_sls, gelu, gemv, gemv_aggregation, histogram, imdb_gt_lt_fp32, imdb_gteq_lt_int64, imdb_lt_int64, imdb_three_col_and, imdb_two_col_and, kmeans_assign, layernorm, memcpy, memset, narrow, relu, residual, softmax, vector_add, vector_exp, wide |
| scalar fp (fadd.s/fsqrt.s/fmadd.s/...) | layernorm, pagerank_inicsr, softmax, spmv |

### Distinct opcodes per workload

| Workload | Distinct opcodes | Vector opcodes |
|---|---|---|
| devmain_self | 21 | 0 |
| dlrm_sls | 28 | 5 |
| gelu | 24 | 12 |
| gemv | 25 | 6 |
| gemv_aggregation | 16 | 5 |
| histogram | 24 | 5 |
| imdb_gt_lt_fp32 | 23 | 8 |
| imdb_gteq_lt_int64 | 26 | 12 |
| imdb_lt_int64 | 23 | 9 |
| imdb_three_col_and | 14 | 4 |
| imdb_two_col_and | 14 | 4 |
| kmeans_assign | 26 | 8 |
| layernorm | 27 | 9 |
| memcpy | 13 | 3 |
| memset | 15 | 3 |
| narrow | 14 | 4 |
| pagerank_inicsr | 20 | 0 |
| relu | 17 | 6 |
| residual | 14 | 4 |
| softmax | 44 | 27 |
| spmv | 21 | 0 |
| sssp | 18 | 0 |
| vector_add | 14 | 4 |
| vector_exp | 34 | 21 |
| wide | 14 | 4 |
