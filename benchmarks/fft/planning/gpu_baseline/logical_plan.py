"""Capture GPU decisions before lowering; never inspect an M2NDP plan.

Traffic is an algorithmic full-array FP32 read/write estimate, excluding
padding, twiddle tables, cache effects and persistent stage launches.
Unextracted GPU launch/LDS parameters are None, never physical substitutes.
"""

from functools import wraps
from inspect import signature
from math import prod


def leaf(source, length, replicas, radices, extra, *, scheme=None, fused=(), upload=None):
    from .common import GPULogicalKernelPlan
    tpt = extra.get("threads_per_transform", extra.get("workers_per_fft"))
    tpb = extra.get("transforms_per_block", extra.get("num_transforms"))
    wgs = extra.get("workgroup_size")
    if tpt is None and wgs is not None and tpb:
        tpt = wgs // tpb
    if wgs is None and tpt is not None and tpb is not None:
        wgs = tpt * tpb
    half = extra.get("half_lds", False)
    return GPULogicalKernelPlan(
        planner_name=source, gpu_scheme=scheme or extra.get("scheme", "Stockham"),
        gpu_kernel_id=0, pass_id=upload if upload is not None else 0,
        fft_length=length, radix_sequence=tuple(radices),
        gpu_workgroup_size=wgs, gpu_threads_per_transform=tpt,
        gpu_transforms_per_workgroup=tpb,
        gpu_shared_memory_bytes=extra.get("gpu_shared_memory_bytes"),
        gpu_fused_operations=tuple(fused), gpu_transpose_role=None,
        gpu_global_read_bytes=length * replicas * 8,
        gpu_global_write_bytes=length * replicas * 8,
        gpu_memory_optimization="half_lds" if half else None,
        gpu_logical_shared_memory_bytes=extra.get("gpu_logical_lds_bytes"),
        m2ndp_memory_optimization_status="NO_EQUIVALENT" if half else None,
        gpu_metadata=tuple((key, extra[key]) for key in ("half_lds", "direct_to_from_reg",
                             "use_3steps_large_twd", "stage_start_size", "axis_swapped", "local_size",
                             "min_registers_per_thread", "registers_per_thread", "registers_per_radix",
                             "block_width", "active_transforms", "block_compute_rounds") if key in extra),
    )


def transpose(source, length, replicas, role, twiddle=False):
    from dataclasses import replace
    return replace(leaf(source, length, replicas, (), {}, scheme="TRANSPOSE",
                        fused=("large_twiddle",) if twiddle else ()), gpu_transpose_role=role)


def _rocfft(args, adapted):
    from . import rocfft_default as roc
    source = "rocfft-default-m2ndp" if adapted else "rocfft-default"
    cu = args["target"].num_ndp_units if adapted else roc.ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT
    n, batch = args["length"], args["batch"]
    match = roc.upstream_sol_map.apply_solution(
        n, placement="op", inverse=args["inverse"], batch=batch,
        in_stride=(1,), out_stride=(1,), i_dist=n, o_dist=n)
    if match is not None:
        raise ValueError("original GPU metadata for a solution-map override has not been extracted")

    def forced(n, r, c, fused=()):
        extra = dict(workgroup_size=c.workgroup_size, threads_per_transform=c.threads_per_transform,
                     transforms_per_block=c.workgroup_size // c.threads_per_transform,
                     scheme=getattr(c, "scheme", "CS_KERNEL_STOCKHAM"),
                     half_lds=getattr(c, "half_lds", False),
                     direct_to_from_reg=getattr(c, "direct_to_from_reg", None),
                     use_3steps_large_twd=getattr(c, "use_3steps_large_twd", None))
        return leaf(source, n, r, c.factors, extra, fused=fused)

    def walk(n, r):
        d = roc.decide_scheme(n, batch=r, multiprocessor_count=cu)
        if d.scheme == "CS_KERNEL_STOCKHAM":
            return [forced(n, r, d.single_kernel)]
        if d.scheme == "CS_BLUESTEIN":
            raise ValueError("rocFFT Bluestein kernel schedule has not been extracted")
        b, a = d.div_length1, n // d.div_length1
        if d.scheme == "CS_L1D_CC":
            return [forced(b, r * a, roc.SBCC_TABLE[b], ("strided_load", "transpose", "large_twiddle")),
                    forced(a, r * b, roc.SBRC_TABLE[a], ("transpose",))]
        return [transpose(source, n, r, "PRE")] + walk(b, r * a) + [
            transpose(source, n, r, "MIDDLE", True), forced(a, r * b, roc.SBRR_TABLE[a]),
            transpose(source, n, r, "POST")]

    return walk(args["length"], args["batch"])


def _vkfft(args):
    from . import vkfft as vk
    n, batch, target = args["length"], args["batch"], args["target"]
    size, cu = args["complex_size_bytes"], args["num_compute_units"]
    factors = vk.schedule_axis_factors(n, target, complex_size_bytes=size)
    source = "vkfft" if cu == 64 else "vkfft-m2ndp"
    entries = []
    for upload in range(len(factors) - 1, -1, -1):
        m, r = factors[upload], batch * n // factors[upload]
        radices = vk.leaf_radix_sequence(m, r, num_compute_units=cu,
                                        num_passes=len(factors), upload_id=upload)
        layout = vk.axisblock_layout_for_leaf(m, radices, max_rhs=r, num_passes=len(factors),
                                      upload_id=upload, original_length=n, target=target,
                                      warp_size=args["warp_size"], num_compute_units=cu,
                                      complex_size_bytes=size, stage_start_size=prod(factors[:upload]))
        decision = vk._direct_radix_plan(m, r, num_compute_units=cu,
                                         num_passes=len(factors), upload_id=upload)
        entries.append(leaf(source, m, r, radices,
                            dict(workers_per_fft=layout.threads_per_transform,
                                 transforms_per_block=layout.transforms_per_block,
                                 axis_swapped=layout.axis_swapped, local_size=layout.local_size,
                                 min_registers_per_thread=decision.min_registers,
                                 registers_per_thread=decision.max_registers,
                                 registers_per_radix=decision.registers_per_radix,
                                 stage_start_size=prod(factors[:upload])),
                            scheme="VkFFT upload", upload=upload,
                            fused=("memory_reordering", "large_twiddle") if len(factors) > 1 and upload > 0
                            else (("memory_reordering",) if len(factors) > 1 else ())))
    return entries


def _clfft(args, name):
    from . import clfft as cl
    adapted = name.endswith("m2ndp")
    source = "clfft-m2ndp" if adapted else "clfft"
    threshold = cl.get_max_1d_length_m2ndp(args["target"]) if adapted else cl.get_max_1d_length()
    batch = args.get("batch", args.get("total_ffts", 1))

    def stockham(n, r):
        c = cl.get_radices(n)
        return leaf(source, n, r, c.radices, dict(workgroup_size=c.workgroup_size,
                                                num_transforms=c.num_transforms))

    def block(n, r, scheme, fused):
        c = cl.get_block_config(n)
        return leaf(source, n, r, c.radices, dict(workgroup_size=c.workgroup_size,
            threads_per_transform=c.threads_per_transform, transforms_per_block=c.block_width,
            gpu_shared_memory_bytes=c.lds_bytes, block_width=c.block_width,
            active_transforms=c.active_transforms, block_compute_rounds=c.compute_rounds),
            scheme=scheme, fused=fused)

    def walk(n, r):
        if cl.is_1d_possible(n, threshold) or name == "plan_single_kernel":
            return [stockham(n, r)]
        a, b = cl.choose_large1d_split(n, threshold)
        if cl.is_block_compute_length(n):
            # Upstream contiguous out-of-place C2C block path integrates
            # the last transpose only when the row factor is <=256.
            first = block(b, r * a, "BCT_C2C", ("strided_load", "transpose", "large_twiddle"))
            second = (block(a, r * b, "BCT_R2C", ("transpose",))
                      if a <= 256 else stockham(a, r * b))
            return [first, second] + ([] if a <= 256 else [transpose(source, n, r, "POST")])
        if a <= 32 and b <= 32:
            from dataclasses import replace
            return [replace(stockham(b, r * a), gpu_fused_operations=("strided_load", "large_twiddle")),
                    replace(stockham(a, r * b), gpu_fused_operations=("strided_store",))]
        return [transpose(source, n, r, "PRE"), stockham(b, r * a),
                transpose(source, n, r, "MIDDLE", True)] + walk(a, r * b) + [transpose(source, n, r, "POST")]

    return walk(args["length"], batch)


def preserve_gpu_plan(planner):
    """Snapshot deterministic source decisions before any physical mapping."""
    @wraps(planner)
    def wrapped(*args, **kwargs):
        from dataclasses import replace
        bound = signature(planner).bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        family = planner.__module__.rsplit(".", 1)[-1]
        from .vkfft import VkfftUnsupportedError
        try:
            if family == "rocfft_default":
                original = _rocfft(values, planner.__name__ == "plan_m2ndp")
            elif family == "vkfft":
                original = _vkfft(values)
            else:
                original = _clfft(values, planner.__name__)
            original = tuple(replace(k, gpu_kernel_id=i,
                                     pass_id=k.pass_id if family == "vkfft" else i)
                             for i, k in enumerate(original))
        except (ValueError, KeyError, VkfftUnsupportedError) as exc:
            original = ()
            capture_error = str(exc)
        else:
            capture_error = None
        result = planner(*args, **kwargs)
        extra = dict(result.gpu_config.extra, original_gpu_plan=original,
                     gpu_plan_capture_error=capture_error)
        return replace(result, gpu_config=replace(result.gpu_config, extra=extra))
    return wrapped
