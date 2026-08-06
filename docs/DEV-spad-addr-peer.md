# Design: `spad_addr` — device_main access to a group's scratchpad

## Motivation

`device_main` orchestrates a task by launching kernels, and a real workload wants
it to read the on-chip state those kernels leave behind. IVF-PQ search
(`benchmarks/ivfpq.mojo`) is the driving case: the natural loop reads a cluster
counter that lives in scratchpad to decide when to stop, and folds every group's
running top-K.

```mojo
def device_main():
    launch_serial[IvfPq.start]()
    while IvfPq.step[0] < NPROBE:          # device_main reads scratchpad to decide flow
        launch_parallel[IvfPq.scan]()
        launch_serial[IvfPq.advance]()
    launch_serial[IvfPq.reduce]()
```

This does not compile today, so the port hard-codes the loop count and pushes the
counter into a kernel. This document proposes the primitive that makes the
direct form legal.

## Addressing by group

Scratchpad is physically per NDP unit, but a workload already thinks in **group
ids** — `group_id()` is the index it maps its data by, ranging over the groups a
launch spreads across. This primitive keeps that vocabulary: it addresses a
**group**'s scratchpad, and maps the group to its owning unit internally. In the
current one-group-per-unit model the group id *is* the unit index; the mapping is
the single place a future many-groups-per-unit layout would change.

To iterate every group (e.g. to fold all of their top-K), `device_main` needs the
count, which no accessor exposes today — the workload threads it through params
(`p.cores`). This design adds it as `num_groups()`, alongside the existing
`group_id()`.

## Why the direct access is blocked (and should stay blocked)

`scratchpad[count, T, name=...]()` emits a **named `addrspace(3)` global**. Any
access to it lowers, in the backend, to `scratchpad_base() + offset`, where
`scratchpad_base` reads the **per-core hardware base register**.

In a kernel that register is correct. In controller code (`device_main`) it holds
the function's own first argument, so a raw scratchpad access reads the wrong
memory. `RISCVM2ndpVerify.cpp` rejects it:

```cpp
// a non-kernel function that computes a scratchpad address is rejected
if (!IsKernel && CB->getIntrinsicID() == Intrinsic::riscv_m2ndp_scratchpad_base)
    reject(F, "the M2NDP scratchpad is a kernel's, and this function is not launched");
```

Because every `addrspace(3)` access lowers through `scratchpad_base`, catching
that one intrinsic catches all raw access. **The check is a correctness guard,
not an arbitrary restriction — it stays.** The primitive below does not loosen
it; it opens one narrow, safe path beside it.

## The primitive

`spad_addr(var, group)` returns the **absolute address** of a scratchpad variable
on a given group. That address decomposes into two pieces, **neither of which
needs `scratchpad_base`**:

```
spad_addr(var, group)  =  region_base(group)            +  offset_of(var)
                          └ M2NDP_SPAD_BASE                └ var's constant offset
                            + unit(group)*(spad_size+guard) ┘  within the .spad block ┘
                       →  inttoptr  →  addrspace(0) pointer   (absolute CXL address)
```

The result is an ordinary `addrspace(0)` pointer holding an absolute CXL address.
Load/store through it take the normal memory path, where the address decoder
(#43) routes it to the owning unit's scratchpad and the NoC timing (#44) charges
the peer round trip. No `addrspace(3)`, no `scratchpad_base`, so the verifier
never fires.

### Contrast

```
kernel:        addrspace(3) access    →  scratchpad_base(per-core) + offset   (base correct)
device_main:   addrspace(3) access    →  ✗ rejected
               spad_addr(var, group)  →  [SPAD_BASE + unit(group)*stride] + offset  →  addrspace(0)  ✓
```

### Mojo surface

`var` is passed as a **compile-time handle** — the `addrspace(3)` pointer that
`scratchpad()` returns, taken symbolically. It is never materialized into a
runtime pointer, since materializing it is exactly what would re-emit
`scratchpad_base` and get rejected.

`handle` is the `addrspace(3)` pointer that `scratchpad()` returns, passed as-is —
just as `__m2ndp_declare_params` takes the params global. It is never accessed
(no `handle[i]`), so it stays a plain global reference the offset call consumes;
the generic base-relative rewrite never touches it.

```mojo
# device_main only: the absolute address of a scratchpad global on `group`.
@always_inline
fn spad_addr[
    T: AnyType, //,
](handle: UnsafePointer[T, address_space = AddressSpace.SHARED], group: Int) -> UnsafePointer[T]:
    var addr = _spad_region_base(group) + _spad_offset(handle)
    return UnsafePointer[UInt8](unsafe_from_address=addr).bitcast[T]()
```

```mojo
def device_main():
    launch_serial[IvfPq.start]()
    while spad_addr(IvfPq.step, 0).load(0) < NPROBE:   # the blocked pattern, now legal
        launch_parallel[IvfPq.scan]()
        launch_serial[IvfPq.advance]()
    for g in range(num_groups()):                      # fold every group's top-K
        var best = spad_addr(IvfPq.top, g).load(0)
        ...
```

## What has to be built

Names follow the existing surface, where the prefix encodes where a symbol is
resolved: plain snake_case is public API (`spad_addr`, `num_groups`,
`spad_capacity`); `__m2ndp_*` is reserved for runtime ABI symbols called through
`external_call` (`__m2ndp_group_id`); a leading underscore is a
compile-time-resolved private mojo helper (`_amo_op_prefix`).

### ① `_spad_offset` — the linchpin

A private mojo helper over `__m2ndp_scratchpad_offset`, a call the scratchpad
layout pass resolves — the same shape as `__m2ndp_declare_params`, which already
exports the params global's offset. It returns the **static offset** of a named
scratchpad global, as a constant, with **no base** — the offset is a compile-time
layout fact, not a runtime call. The `.spad` section has origin 0, so the global's
link-time address *is* its offset.

- `RISCVM2ndpLowerScratchpad` collects `__m2ndp_scratchpad_offset(@var)` calls,
  replaces each with the constant offset, and **skips its use of the global** in
  the generic `addrspace(3) → scratchpad_base + offset` rewrite.
- So no `scratchpad_base` is emitted for it, and at verify time the kernel-only
  check passes untouched.

Implemented on the LLVM fork branch `m2ndp/spad-offset`.

### ② `_spad_region_base(group)` — the per-group base

`M2NDP_SPAD_BASE + unit(group) * (spad_capacity() + guard)`, where `unit(group)`
maps a group to its owning NDP unit (identity in the current model).
`M2NDP_SPAD_BASE` and the guard size are fixed in `address_map.h`; the per-unit
scratchpad size comes from the **runtime-selected config** via `spad_capacity()`
(below) — the same value the simulator lays the regions out with, so device and
sim agree by construction. This is the mojo counterpart of the simulator's
`AddressDecoder::scratchpad_base(unit)`.

### ③ `num_groups()` — the group count

A public accessor returning the number of groups the task spreads across, so
`device_main` can iterate them without threading a count through params. It
mirrors `group_id()`: a snake_case wrapper over a runtime ABI symbol
`__m2ndp_num_groups`, seeded by the controller from the launch geometry.

### ④ `spad_capacity()` — the per-unit scratchpad size

A public accessor returning the size, in bytes, of one unit's scratchpad, taken
from the runtime-selected config. It mirrors `group_id()`: a snake_case wrapper
over a runtime ABI symbol `__m2ndp_spad_capacity`, seeded by the controller.
`_spad_region_base` uses it for the region stride, and a workload can size its own
buffers against it.

### ⑤ Result pointer — reuse the existing path

`inttoptr` to a plain `addrspace(0)` `UnsafePointer[T]`. Load/store already route
through the address decoder (#43) and are timed over the NoC (#44). Nothing new
here.

## Semantics and open decisions

- **Coherence.** A load/store through the returned pointer is a peer access: its
  value is read/written at access time, and the NoC round trip sets the timing
  (per #44). Cross-unit ordering (a barrier) is deferred, as in #44.
- **Group range.** `group ∈ [0, num_groups())`; out of range faults.
- **Kernels may use it too.** The same primitive from a kernel forms a cross-group
  peer access — exactly what #44 models. It is not restricted to `device_main`.

## Dependencies and acceptance

- **Routing and timing already exist**: address decoder #43 (`spad-at-primitive`),
  peer NoC timing #44 (`remote-spad-timing`), both on top of #42
  (`spad-ownership-hoist`). This design adds the compiler/mojo front end that lets
  `device_main` form such an address legally.
- **Acceptance test**: a device_main that reads a group's counter through
  `spad_addr` and drives a data-dependent loop over `num_groups()`; then rewrite
  `benchmarks/ivfpq.mojo` to the direct form and promote its manifest row from
  `xfail` to `pass`.
