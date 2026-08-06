# Task primitives

The vocabulary a workload uses inside an `NDPTask`: launching kernels, finding its
place in the data, combining results across µthreads, and reaching scratchpad.
This is the workload author's reference.

> Backend/ABI contract: [`INTERFACE.md`](INTERFACE.md). Worked workloads:
> [`EXAMPLES.md`](EXAMPLES.md). Scratchpad-access rationale:
> [`DEV-spad-addr-peer.md`](DEV-spad-addr-peer.md).

## Launch — `device_main`

`device_main` runs the task by launching kernels. A launch is **synchronous**: the
call returns only once the kernel has finished, so the next stage sees what the
previous one wrote.

| Primitive | Meaning |
|-----------|---------|
| `launch_parallel[kernel]()` | run `kernel` across every group |
| `launch_serial[kernel]()` | run `kernel` on a single group |

```mojo
def device_main():
    launch_parallel[MyTask.compute]()
    launch_serial[MyTask.reduce]()
```

## Identity and geometry

A kernel finds the data it was mapped to from its identity; the geometry accessors
read the task's runtime config, so a workload need not thread the numbers through
its params.

| Primitive | Where | Meaning |
|-----------|-------|---------|
| `local_uthread_id() -> Int` | kernel | index among the µthreads on this core |
| `global_uthread_id() -> Int` | kernel | index across all cores; identifies the mapped data |
| `group_id() -> Int` | kernel | which group this µthread belongs to |
| `num_groups() -> Int` | kernel, device_main | how many groups the task spreads across |
| `spad_capacity() -> Int` | kernel, device_main | bytes of scratchpad on one unit |

```mojo
var g = group_id()
for i in range(global_uthread_id(), N, num_groups()):
    ...
```

## Atomics — kernel

With no barrier, atomics are how µthreads combine results. They lower to
`atomicrmw` (relaxed ordering) and work on ordinary memory and on scratchpad.

| Primitive | Meaning |
|-----------|---------|
| `atomic_add(ptr, val)` | add, returning the previous value |
| `atomic_max(ptr, val)` | max, returning the previous value |
| `atomic_add_lanes(ptr, vec)` | one scalar atomic per lane into `ptr` |
| `atomic_add_indexed(base, idx, vec)` | per-lane atomic, each lane to its own address |

```mojo
_ = atomic_add(counter, Int32(1))
```

## Scratchpad

Scratchpad is the per-unit on-chip memory a task keeps its hot state in.

**Declare** a named buffer once, as a task-level `comptime` — every µthread on the
unit shares it:

```mojo
struct IvfPq(NDPTask):
    comptime top  = scratchpad[2 * TOPK, Float32, name="ivfpq_top"]()
    comptime step = scratchpad[1, Int32, name="ivfpq_step"]()
```

**In a kernel**, index it directly — it is this unit's own scratchpad:

```mojo
IvfPq.step[0] += 1
```

**From `device_main`** (which has no scratchpad of its own) you **cannot** index a
scratchpad variable directly — that is a compile error, on purpose. Use
`spad_addr`, which returns the **address** of a variable in a given group's
scratchpad as a normal pointer:

| Primitive | Meaning |
|-----------|---------|
| `spad_addr(var, group) -> UnsafePointer[T]` | address of `var` in `group`'s scratchpad |

```mojo
def device_main():
    launch_serial[IvfPq.start]()
    while spad_addr(IvfPq.step, 0).load(0) < NPROBE:   # read group 0's counter to drive the loop
        launch_parallel[IvfPq.scan]()
        launch_serial[IvfPq.advance]()
    for g in range(num_groups()):                      # fold every group's top-K
        var best = spad_addr(IvfPq.top, g).load(0)
        ...
```

A kernel can use `spad_addr` too, to reach another group's scratchpad; reaching
its own group is just the direct `var[i]` form.

## Notes

- **Peer cost.** A `spad_addr` access to another group crosses the on-chip
  network and is charged the round trip; a kernel touching its own scratchpad is
  the fast path.
- **Ordering.** Launches are synchronous, but across groups within a stage there
  is no barrier — read another group's scratchpad only where its writer has
  already completed.
- **Ranges.** `group ∈ [0, num_groups())`; out of range faults.
