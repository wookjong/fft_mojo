from __future__ import annotations

"""One place for the hardware/simulator constants this PoC otherwise
duplicates wherever a planner needs them -- there is no single source in
this repo both the C++ simulator config and this Python planning code could
read them from (see each field's own comment for its concrete upstream
source), so `DEFAULT_TARGET_PROFILE` is the one place a value gets typed in
and every planner/generator call site threads the same `TargetProfile`
through instead of re-declaring its own module constant.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProfile:
    lmul1_float32_lanes: int
    spad_capacity_bytes: int
    max_concurrent_scratchpad_bytes: int
    interleave_chunk_uthreads: int


DEFAULT_TARGET_PROFILE = TargetProfile(
    # This target's guaranteed vector register length (+zvl128b, see
    # scripts/build.sh's FEATURES) in Float32 elements: 128 bits / 32 bits.
    # Above this width RVV needs LMUL>1 register grouping, which is exactly
    # what makes a heavily-unrolled kernel spill (see make_fft_kernel's own
    # `compute_lanes` discussion and docs/STATUS.md).
    lmul1_float32_lanes=4,
    # One NDP unit's own scratchpad, bytes -- matches `spad_size` in
    # third_party/m2ndp-detour/config/performance/M2NDP/m2ndp.config and
    # `spad (rw) : ORIGIN = 0, LENGTH = 128K` in scripts/m2ndp.lds (both
    # 131072). A margin below the real 131072 (rather than that exact
    # figure) leaves room for a kernel's own Params struct and other
    # scratchpad-resident globals (see scripts/m2ndp.lds's own docstring:
    # "the globals are laid out from" the scratchpad base), which
    # _cap_max_uthread has no visibility into and so cannot budget for
    # itself.
    spad_capacity_bytes=120 * 1024,
    # A cap on `max_uthread * bytes_per_uthread` -- how many bytes of
    # scratchpad may be concurrently active across every uthread resident on
    # one NDP unit at once, independent of how many total bytes
    # spad_capacity_bytes alone would allow. Found by hand on N=8192's
    # FFTRecNear0 (4096 bytes/uthread -- length 256, ping-pong-doubled: see
    # fft_plan_core._build_plan's own `bytes_per_uthread = len(buffer_names)
    # * scratchpad_stride * 4`, not FFTCodegenPlan.scratchpad_uthread_stride
    # alone, which is only one buffer's share): 16 concurrent uthreads
    # (65536 bytes) finishes in ~90,000 simulated cycles; 30 (122880 bytes)
    # blew *past* the simulator's fixed 20,000,000-cycle-per-launch budget
    # for the exact same kernel and data. 16 * 4096 = 65536 is the largest
    # *confirmed-safe* point on that line, so that's the budget here -- not
    # a fitted formula. A flat uthread-*count* cap (always 16) was tried
    # first and also fixes this, but then wrongly re-caps kernels whose own
    # footprint was never at risk (e.g. N=1024's FFTRecLeaf1 at 64 bytes/
    # uthread, fine at 256 concurrent uthreads -- 16384 bytes total),
    # forcing them into extra small launches that only add per-launch
    # overhead. Capping the byte product instead leaves those uncapped
    # while still catching Near0-shaped kernels at any N. See
    # _cap_max_uthread's own docstring.
    max_concurrent_scratchpad_bytes=16 * 4096,
    # The M2NDP address decoder hands consecutive microthreads to the same
    # NDP unit in blocks of this size before rotating to the next unit (the
    # real config's `m2ndp_interleave_size / packet_size`). A cooperative
    # leaf's `workers_per_fft` must divide this so that local_uthread_id()'s
    # and global_uthread_id()'s own groupings of `workers_per_fft` partition
    # the same physical microthreads into the same groups -- see
    # fft_codegen._emit_cooperative_stage's own docstring for the concrete
    # trace (against this exact config) that found this the hard way.
    interleave_chunk_uthreads=8,
)
