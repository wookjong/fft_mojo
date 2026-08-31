from __future__ import annotations

"""Numeric + index-only verification for fft_plan_recursive.py's
make_recursive_transpose_plan: the generic tiled-transpose's own bijection
+ DRAM access-shape check (square/rectangular, full tile, every tail
combination, multiple replicas), a whole-tree index-only walk (for N too
large for the numeric path to stay fast -- also confirms every MIDDLE
transpose's own twiddle_modulus equals *that node's own* m, never a
stale/global value), and full numeric re-execution of the actual emitted
stage text -- both FFT leaf kernels (verify_fft_harness.run_kernel) and
PRE/MIDDLE/POST transpose kernels (this module's own run_physical_transpose)
-- against numpy.fft/ifft.
"""

import types

import numpy as np

from planning.fft_plan_recursive import (
    FFTLeafPlan,
    FFTRecursiveNodePlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
    make_recursive_transpose_plan,
)
from codegen.fft_codegen import Emitter
from codegen.fft_transpose_codegen import _emit_physical_transpose_stage
from verification.verify_fft_harness import Ptr, _simd, _translate_emitted_lines, run_kernel
from verification.verify_fft_persistent import run_persistent_kernel


def _translate_physical_transpose_stage(plan: PhysicalTransposePlan) -> str:
    """Same discipline as _translate_stage/_translate_transpose_stage:
    re-execute the actual text fft_transpose_codegen.py emits."""
    e = Emitter()
    _emit_physical_transpose_stage(e, plan=plan)
    return _translate_emitted_lines(e.lines)


def run_physical_transpose(
    plan: PhysicalTransposePlan, *, src_real: Ptr, src_imag: Ptr,
    dst_real: Ptr, dst_imag: Ptr, tw_real: Ptr | None = None, tw_imag: Ptr | None = None,
) -> None:
    num_groups = -(-plan.total_uthreads // plan.max_uthread)
    group_namespaces = [
        types.SimpleNamespace(tile_buf=Ptr(plan.scratchpad_elements * plan.max_uthread))
        for _ in range(num_groups)
    ]
    p_ns = types.SimpleNamespace(
        src_real_base=src_real, src_imag_base=src_imag,
        dst_real_base=dst_real, dst_imag_base=dst_imag,
    )
    if plan.twiddle_modulus is not None:
        assert tw_real is not None and tw_imag is not None
        p_ns.twiddle_real_base = tw_real
        p_ns.twiddle_imag_base = tw_imag
    # Same tables _emit_physical_transpose_stage reads at runtime when a
    # non-power-of-2 tiles_per_replica/grid_cols would otherwise need a
    # `mulhsu`-lowering `//`/`%` (see PhysicalTransposePlan.needs_q_table/
    # needs_tr_table) -- filled the same way generate_recursive_fft_kernels's
    # own host main() does, plain Python `//` here since this harness never
    # touches the simulator at all.
    if plan.needs_q_table:
        tiles_per_replica = plan.grid_rows * plan.grid_cols
        q_table = Ptr(plan.total_uthreads)
        for tid in range(plan.total_uthreads):
            q_table[tid] = tid // tiles_per_replica
        p_ns.q_table = q_table
    if plan.needs_tr_table:
        tiles_per_replica = plan.grid_rows * plan.grid_cols
        t_r_table = Ptr(tiles_per_replica)
        for lt in range(tiles_per_replica):
            t_r_table[lt] = lt // plan.grid_cols
        p_ns.t_r_table = t_r_table
    if plan.needs_round_split:
        # On real hardware this is generate_recursive_fft_kernels' own
        # per-round host constant (`r * plan.max_uthread`, added once so
        # `tile_id` stays absolute across a stage split into several
        # `.launch()` calls -- see the round loop there). This harness
        # runs every uthread in one Python pass over the *whole*
        # 0..total_uthreads-1 range instead of one pass per round, so
        # `global_uthread_id()` is already absolute on its own -- 0 leaves
        # `tile_id = global_uthread_id() + p.round_offset` exactly the
        # `tile_id = global_uthread_id()` every non-round-split plan
        # computes, only present here because the emitted Params struct
        # always declares the field once `needs_round_split` is true (see
        # _emit_physical_transpose_params_struct), whether or not this
        # particular translate/exec call cares about rounds.
        p_ns.round_offset = 0
    current = {"global_id": 0, "local_id": 0}
    src = _translate_physical_transpose_stage(plan)
    namespace = {
        "Float32": float,
        "SIMD": _simd,
        "local_uthread_id": lambda: current["local_id"],
        "global_uthread_id": lambda: current["global_id"],
        f"MAX_UTHREAD_{plan.kernel_name}": plan.max_uthread,
        "p": p_ns,
    }
    code = compile(src, f"<{plan.kernel_name} stage 0>", "exec")
    exec(code, namespace)
    stage_fn = namespace["stage_0"]
    for global_id in range(plan.total_uthreads):
        current["global_id"] = global_id
        current["local_id"] = global_id % plan.max_uthread
        namespace[plan.kernel_name] = group_namespaces[global_id // plan.max_uthread]
        stage_fn()


def _make_physical_twiddle_table(plan: PhysicalTransposePlan) -> tuple[np.ndarray, np.ndarray]:
    """Independent host-side fill (re-derives the angle itself, not calling
    into fft_transpose_codegen's own precompute) for a MIDDLE transpose's
    dense W_M table -- table[r*cols+c] = W_M^(r*c), modulus = this node's
    own twiddle_modulus (never the top-level N for a non-root node)."""
    assert plan.twiddle_modulus is not None
    real = np.zeros(plan.rows * plan.cols)
    imag = np.zeros(plan.rows * plan.cols)
    sign = 1.0 if plan.inverse else -1.0
    for r in range(plan.rows):
        for c in range(plan.cols):
            angle = sign * 2.0 * np.pi * r * c / plan.twiddle_modulus
            real[r * plan.cols + c] = np.cos(angle)
            imag[r * plan.cols + c] = np.sin(angle)
    return real, imag


def verify_physical_transpose_shape(plan: PhysicalTransposePlan) -> dict[str, int]:
    """Index-only (no floats): every source/destination address is visited
    exactly once (full tile, row tail, column tail, both tails, multiple
    replicas all naturally exercised depending on plan), and every DRAM
    vector load/store has elem_stride==1 (checked directly: each row's
    address range is a contiguous run, by the same formula codegen uses,
    independently re-derived here)."""
    rows, cols = plan.rows, plan.cols
    tile_rows, tile_cols = plan.tile_rows, plan.tile_cols
    grid_rows, grid_cols = plan.grid_rows, plan.grid_cols
    seen_src: set[int] = set()
    seen_dst: set[int] = set()
    max_src_row_jump = 0
    max_dst_row_jump = 0
    tiles_per_replica = grid_rows * grid_cols
    for tile_id in range(plan.total_uthreads):
        q = tile_id // tiles_per_replica
        local = tile_id % tiles_per_replica
        t_r = local // grid_cols
        t_c = local % grid_cols
        valid_r = min(tile_rows, rows - t_r * tile_rows)
        valid_c = min(tile_cols, cols - t_c * tile_cols)

        row_bases = []
        for i in range(valid_r):
            r = t_r * tile_rows + i
            base = q * rows * cols + r * cols + t_c * tile_cols
            for off in range(valid_c):
                addr = base + off
                if addr in seen_src:
                    raise AssertionError(f"{plan.kernel_name}: src addr {addr} visited twice")
                seen_src.add(addr)
            row_bases.append(base)
        if len(row_bases) > 1:
            max_src_row_jump = max(max_src_row_jump, max(row_bases[k + 1] - row_bases[k] for k in range(len(row_bases) - 1)))

        col_bases = []
        for j in range(valid_c):
            c = t_c * tile_cols + j
            base = q * rows * cols + c * rows + t_r * tile_rows
            for off in range(valid_r):
                addr = base + off
                if addr in seen_dst:
                    raise AssertionError(f"{plan.kernel_name}: dst addr {addr} visited twice")
                seen_dst.add(addr)
            col_bases.append(base)
        if len(col_bases) > 1:
            max_dst_row_jump = max(max_dst_row_jump, max(col_bases[k + 1] - col_bases[k] for k in range(len(col_bases) - 1)))

    total = plan.replica_count * rows * cols
    if seen_src != set(range(total)):
        raise AssertionError(f"{plan.kernel_name}: src addresses are not a permutation of 0..{total - 1}")
    if seen_dst != set(range(total)):
        raise AssertionError(f"{plan.kernel_name}: dst addresses are not a permutation of 0..{total - 1}")
    return {
        "load_elem_stride": 1, "store_elem_stride": 1,
        "max_src_row_jump": max_src_row_jump, "max_dst_row_jump": max_dst_row_jump,
    }


def verify_recursive_tree_index_only(node: FFTLeafPlan | FFTRecursiveNodePlan) -> None:
    """Index-only, suitable for N too large for the numeric path: walks
    every PRE/MIDDLE/POST transpose in the tree and checks bijection +
    access shape; also confirms MIDDLE's own twiddle_modulus always equals
    *that node's own* m, never a stale/global value."""
    if isinstance(node, FFTLeafPlan):
        return
    assert isinstance(node, FFTRecursiveNodePlan)
    verify_physical_transpose_shape(node.pre_transpose)
    verify_physical_transpose_shape(node.middle_transpose)
    if node.middle_transpose.twiddle_modulus != node.m:
        raise AssertionError(
            f"MIDDLE transpose {node.middle_transpose.kernel_name}: twiddle_modulus="
            f"{node.middle_transpose.twiddle_modulus} != this node's own m={node.m}"
        )
    verify_recursive_tree_index_only(node.far_child)
    verify_physical_transpose_shape(node.post_transpose)


def run_recursive_plan(
    plan: RecursiveFFTPlan, x: np.ndarray, *,
    compute_lanes: int | None = None, narrow_middle_stages: bool = False,
    loop_stages: bool = True,
) -> np.ndarray:
    """Full numeric chain: every stage's *actual emitted* text is
    translated and re-executed (run_kernel for FFT leaves,
    run_physical_transpose for PRE/MIDDLE/POST transposes) -- same
    discipline as every other run_* helper in this file.

    `compute_lanes`: threaded straight through to `run_kernel` for every
    FFT-leaf stage (`None`, the default, matches `run_kernel`'s own
    default) -- see that function's own docstring. PhysicalTransposePlan
    stages are untouched either way, matching generate_recursive_fft_
    kernels's own scoping of this parameter to `_emit_kernel` only.

    `loop_stages` defaults to `True` to match
    generate_recursive_fft_kernels's own default (see
    fft_transpose_codegen.generate_recursive_fft_kernels): the actual
    kernels this strategy emits render looped stages unless a caller opts
    out, so verifying with the unrolled (`False`) path by default would
    leave the runtime-loop rendering (`_try_build_loop_stage`/
    `_emit_loop_stage`) numerically unchecked.
    """
    n = plan.n
    stages = flatten_recursive_node(plan.root)
    cur_r, cur_i = Ptr(n), Ptr(n)
    cur_r.arr[:] = x.real
    cur_i.arr[:] = x.imag
    for stage in stages:
        next_r, next_i = Ptr(n), Ptr(n)
        if isinstance(stage, PhysicalTransposePlan):
            if stage.twiddle_modulus is not None:
                tw_re, tw_im = _make_physical_twiddle_table(stage)
                tw_r, tw_i = Ptr(len(tw_re)), Ptr(len(tw_re))
                tw_r.arr[:] = tw_re
                tw_i.arr[:] = tw_im
                run_physical_transpose(stage, src_real=cur_r, src_imag=cur_i, dst_real=next_r, dst_imag=next_i, tw_real=tw_r, tw_imag=tw_i)
            else:
                run_physical_transpose(stage, src_real=cur_r, src_imag=cur_i, dst_real=next_r, dst_imag=next_i)
        elif stage.persistent is not None:
            # A persistent leaf cannot go through run_kernel at all (see
            # verify_fft_persistent.run_persistent_kernel's own docstring:
            # a different round/group model entirely, not just a
            # different worker-dispatch flavor of the same one) --
            # num_logical_blocks is this leaf's own replica count, exactly
            # what fft_plan_recursive._build_leaf_kernel passed through as
            # total_uthreads/num_logical_blocks when it built this plan.
            assert stage.host.total_elems % stage.length == 0
            num_logical_blocks = stage.host.total_elems // stage.length
            run_persistent_kernel(
                stage, num_logical_blocks=num_logical_blocks,
                input_real=cur_r, input_imag=cur_i, output_real=next_r, output_imag=next_i,
                compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            )
        else:
            assert stage.large_twiddle is None
            # A cooperative stage never loops regardless of the caller's own
            # `loop_stages` -- see generate_recursive_fft_kernels's own
            # identical per-stage decision (fft_transpose_codegen.py) for why.
            run_kernel(
                stage, input_real=cur_r, input_imag=cur_i, output_real=next_r, output_imag=next_i,
                compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
                loop_stages=loop_stages and stage.cooperation is None,
            )
        cur_r, cur_i = next_r, next_i
    return cur_r.arr + 1j * cur_i.arr


def verify_recursive_plan(
    n: int, *, scratchpad_byte_budget: int, inverse: bool, seed: int,
    tile_rows: int | None = None, tile_cols: int | None = None,
    compute_lanes: int | None = None, narrow_middle_stages: bool = False,
    loop_stages: bool = True,
    spad_capacity_bytes: int | None = None,
    max_concurrent_scratchpad_bytes: int | None = None,
    cooperative_workers: int | str | None = None,
    forced_worker_sequence: tuple[int | str | None, ...] | None = None,
) -> tuple[float, RecursiveFFTPlan]:
    """`compute_lanes`/`narrow_middle_stages`: threaded straight through to
    `run_recursive_plan` -- see that function's and `run_kernel`'s own
    docstrings. `None`/`False` (the defaults) match their own defaults,
    unchanged from before these parameters existed.

    `forced_worker_sequence`: threaded straight through to
    `make_recursive_transpose_plan` -- see `_build_recursive_node`'s own
    docstring (mutually exclusive with `cooperative_workers`). `None` (the
    default) is unchanged from before this parameter existed.

    `spad_capacity_bytes`/`max_concurrent_scratchpad_bytes`: both `None`
    by default (unchanged from before either existed) -- pass a
    `max_concurrent_scratchpad_bytes` small enough to force some kernel's
    own `max_uthread < total_uthreads` to numerically exercise
    generate_recursive_fft_kernels' round-split launches (see
    _cap_max_uthread/PhysicalTransposePlan.needs_round_split). This only checks the *stage
    body*'s own address formula stays correct once max_uthread is capped
    (global_uthread_id() * row_stride + base, same value whether summed as
    one absolute range here or as round*max_uthread + a per-round-relative
    id against a round-shifted base pointer on real hardware -- the two
    are algebraically identical) -- run_kernel/run_physical_transpose only
    ever re-execute a stage's own emitted text, never
    generate_recursive_fft_kernels' own host-level round loop /
    round_offset-add / pointer-offset arithmetic that decides *how many*
    rounds a stage's launch actually splits into, so a bug confined to
    that host-level loop is invisible here regardless of this parameter --
    only the real Mojo -> llc -> M2NDP-Detour toolchain (run_fft_test.sh)
    re-executes that.
    """
    plan = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=scratchpad_byte_budget, inverse=inverse,
        tile_rows=tile_rows, tile_cols=tile_cols,
        spad_capacity_bytes=spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
        cooperative_workers=cooperative_workers,
        forced_worker_sequence=forced_worker_sequence,
    )
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    got = run_recursive_plan(
        plan, x, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        loop_stages=loop_stages,
    )
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected))), plan


def summarize_recursive_plan(plan: RecursiveFFTPlan) -> None:
    """Debug/summary output: per-node PRE/FFT(B)/MIDDLE+twiddle/child/POST
    shapes, then whole-plan totals (recursion depth, leaf/transpose kernel
    counts, total tiles, max per-tile scratchpad bytes, DRAM full-array
    passes, max DRAM elem_stride). elem_stride here is always 1 by
    construction (every transpose is tile-based) -- printed to make that
    visible, not because it might vary."""

    def describe(node, depth: int) -> None:
        pad = "  " * depth
        if isinstance(node, FFTLeafPlan):
            print(f"{pad}Leaf M={node.m} R={node.r}: {node.kernel.kernel_name} radices={tuple(s.radix for s in node.kernel.stages)}")
            return
        print(f"{pad}Node M={node.m} R={node.r}  split: A={node.a} B={node.b}")
        pt = node.pre_transpose
        print(f"{pad}  PRE    matrix={pt.rows}x{pt.cols} tile={pt.tile_rows}x{pt.tile_cols} tiles={pt.total_uthreads} elem_stride=1")
        print(f"{pad}  FFT(B) {node.near_fft.kernel.kernel_name} M={node.near_fft.m} R={node.near_fft.r}")
        mt = node.middle_transpose
        print(f"{pad}  MIDDLE matrix={mt.rows}x{mt.cols} twiddle_modulus={mt.twiddle_modulus} tile={mt.tile_rows}x{mt.tile_cols} tiles={mt.total_uthreads} elem_stride=1")
        describe(node.far_child, depth + 1)
        pot = node.post_transpose
        print(f"{pad}  POST   matrix={pot.rows}x{pot.cols} tile={pot.tile_rows}x{pot.tile_cols} tiles={pot.total_uthreads} elem_stride=1 apply_1/N={pot.apply_inverse_scale}")

    print(f"N = {plan.n}")
    describe(plan.root, 0)

    stages = flatten_recursive_node(plan.root)
    leaf_count = sum(1 for s in stages if not isinstance(s, PhysicalTransposePlan))
    transpose_count = sum(1 for s in stages if isinstance(s, PhysicalTransposePlan))
    total_tiles = sum(s.total_uthreads for s in stages if isinstance(s, PhysicalTransposePlan))
    max_spad_bytes = max(
        (s.scratchpad_elements * 4 for s in stages if isinstance(s, PhysicalTransposePlan)),
        default=0,
    )
    print()
    print(f"  leaf FFT kernel count:     {leaf_count}")
    print(f"  transpose kernel count:    {transpose_count}")
    print(f"  total transpose tiles:     {total_tiles}")
    print(f"  max tile scratchpad bytes: {max_spad_bytes}")
    print(f"  DRAM full-array passes:    {len(stages)} (one input read + one output write per stage)")
    print(f"  max DRAM elem_stride:      1 (every stage here is either an FFT leaf's own contiguous mapping or a tiled transpose)")

