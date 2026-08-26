from __future__ import annotations

"""Radix-local FFT butterfly code generation for M2NDP.

This module deliberately knows nothing about scratchpad layout, stage twiddles,
or DRAM stores.  It only transforms already-loaded SIMD vectors

    rr0..rr(R-1), ii0..ii(R-1)

into

    or0..or(R-1), oi0..oi(R-1)

for one radix-R transform in every SIMD lane.

The main FFT code generator owns all memory/layout decisions.  Keeping the
radix algebra here makes it easy to replace one butterfly without touching the
stage scheduler.
"""

from math import cos, pi, sin
from typing import Callable, Protocol

from codegen.common import f32 as _f32
from radix_spec import SUPPORTED_RADICES


class LineEmitter(Protocol):
    def add(self, line: str = "") -> None: ...


# Called with an output index k as soon as or{k}/oi{k} are fully computed --
# lets the caller (fft_codegen._emit_batch) interleave that output's
# twiddle+store right there, instead of holding every output of a
# many-output radix (6/8/9/10/16 factorized; 7/11/13/17 symmetric) live
# until the whole butterfly returns. Radix 2/3/4/5 compute all their
# outputs together regardless (see _emit_small_named), so calling it in a
# plain loop after the fact is unchanged behavior for those.
OutputCallback = Callable[[int], None]


def _out(prefix: str, k: int) -> tuple[str, str]:
    return f"{prefix}r{k}", f"{prefix}i{k}"


def _emit_complex_twiddle(
    e: LineEmitter,
    *,
    indent: str,
    src_r: str,
    src_i: str,
    dst_r: str,
    dst_i: str,
    exponent: int,
    modulus: int,
    inverse: bool,
) -> None:
    """Emit multiplication by W_modulus**exponent with constants folded.

    Special rotations 1, -1, +/-j are emitted without real multiplications.
    """
    sign = 1.0 if inverse else -1.0
    angle = sign * 2.0 * pi * exponent / modulus
    wr = cos(angle)
    wi = sin(angle)

    eps = 1.0e-12
    if abs(wi) < eps and abs(wr - 1.0) < eps:
        e.add(f"{indent}var {dst_r} = {src_r}")
        e.add(f"{indent}var {dst_i} = {src_i}")
    elif abs(wi) < eps and abs(wr + 1.0) < eps:
        e.add(f"{indent}var {dst_r} = -{src_r}")
        e.add(f"{indent}var {dst_i} = -{src_i}")
    elif abs(wr) < eps and abs(wi - 1.0) < eps:
        # +j * (r + ji) = -i + jr
        e.add(f"{indent}var {dst_r} = -{src_i}")
        e.add(f"{indent}var {dst_i} = {src_r}")
    elif abs(wr) < eps and abs(wi + 1.0) < eps:
        # -j * (r + ji) = i - jr
        e.add(f"{indent}var {dst_r} = {src_i}")
        e.add(f"{indent}var {dst_i} = -{src_r}")
    else:
        e.add(
            f"{indent}var {dst_r} = {src_r} * {_f32(wr)} - {src_i} * {_f32(wi)}"
        )
        e.add(
            f"{indent}var {dst_i} = {src_r} * {_f32(wi)} + {src_i} * {_f32(wr)}"
        )


def _emit_radix2_named(
    e: LineEmitter,
    *,
    indent: str,
    ins: list[tuple[str, str]],
    prefix: str,
) -> list[tuple[str, str]]:
    (a_r, a_i), (b_r, b_i) = ins
    o0r, o0i = _out(prefix, 0)
    o1r, o1i = _out(prefix, 1)
    e.add(f"{indent}var {o0r} = {a_r} + {b_r}")
    e.add(f"{indent}var {o0i} = {a_i} + {b_i}")
    e.add(f"{indent}var {o1r} = {a_r} - {b_r}")
    e.add(f"{indent}var {o1i} = {a_i} - {b_i}")
    return [(o0r, o0i), (o1r, o1i)]


def _emit_radix3_named(
    e: LineEmitter,
    *,
    indent: str,
    ins: list[tuple[str, str]],
    prefix: str,
    inverse: bool,
) -> list[tuple[str, str]]:
    (r0, i0), (r1, i1), (r2, i2) = ins
    c = _f32(0.5)
    s = _f32(3.0**0.5 / 2.0)

    e.add(f"{indent}var {prefix}sum12r = {r1} + {r2}")
    e.add(f"{indent}var {prefix}sum12i = {i1} + {i2}")
    e.add(f"{indent}var {prefix}diff12r = {r1} - {r2}")
    e.add(f"{indent}var {prefix}diff12i = {i1} - {i2}")
    e.add(f"{indent}var {prefix}baser = {r0} - {prefix}sum12r * {c}")
    e.add(f"{indent}var {prefix}basei = {i0} - {prefix}sum12i * {c}")

    o0r, o0i = _out(prefix, 0)
    o1r, o1i = _out(prefix, 1)
    o2r, o2i = _out(prefix, 2)
    e.add(f"{indent}var {o0r} = {r0} + {prefix}sum12r")
    e.add(f"{indent}var {o0i} = {i0} + {prefix}sum12i")
    if inverse:
        e.add(f"{indent}var {o1r} = {prefix}baser - {prefix}diff12i * {s}")
        e.add(f"{indent}var {o1i} = {prefix}basei + {prefix}diff12r * {s}")
        e.add(f"{indent}var {o2r} = {prefix}baser + {prefix}diff12i * {s}")
        e.add(f"{indent}var {o2i} = {prefix}basei - {prefix}diff12r * {s}")
    else:
        e.add(f"{indent}var {o1r} = {prefix}baser + {prefix}diff12i * {s}")
        e.add(f"{indent}var {o1i} = {prefix}basei - {prefix}diff12r * {s}")
        e.add(f"{indent}var {o2r} = {prefix}baser - {prefix}diff12i * {s}")
        e.add(f"{indent}var {o2i} = {prefix}basei + {prefix}diff12r * {s}")
    return [(o0r, o0i), (o1r, o1i), (o2r, o2i)]


def _emit_radix4_named(
    e: LineEmitter,
    *,
    indent: str,
    ins: list[tuple[str, str]],
    prefix: str,
    inverse: bool,
) -> list[tuple[str, str]]:
    (r0, i0), (r1, i1), (r2, i2), (r3, i3) = ins
    e.add(f"{indent}var {prefix}a0r = {r0} + {r2}")
    e.add(f"{indent}var {prefix}a0i = {i0} + {i2}")
    e.add(f"{indent}var {prefix}a1r = {r0} - {r2}")
    e.add(f"{indent}var {prefix}a1i = {i0} - {i2}")
    e.add(f"{indent}var {prefix}b0r = {r1} + {r3}")
    e.add(f"{indent}var {prefix}b0i = {i1} + {i3}")
    e.add(f"{indent}var {prefix}b1r = {r1} - {r3}")
    e.add(f"{indent}var {prefix}b1i = {i1} - {i3}")

    outs = [_out(prefix, k) for k in range(4)]
    e.add(f"{indent}var {outs[0][0]} = {prefix}a0r + {prefix}b0r")
    e.add(f"{indent}var {outs[0][1]} = {prefix}a0i + {prefix}b0i")
    e.add(f"{indent}var {outs[2][0]} = {prefix}a0r - {prefix}b0r")
    e.add(f"{indent}var {outs[2][1]} = {prefix}a0i - {prefix}b0i")
    if inverse:
        e.add(f"{indent}var {outs[1][0]} = {prefix}a1r - {prefix}b1i")
        e.add(f"{indent}var {outs[1][1]} = {prefix}a1i + {prefix}b1r")
        e.add(f"{indent}var {outs[3][0]} = {prefix}a1r + {prefix}b1i")
        e.add(f"{indent}var {outs[3][1]} = {prefix}a1i - {prefix}b1r")
    else:
        e.add(f"{indent}var {outs[1][0]} = {prefix}a1r + {prefix}b1i")
        e.add(f"{indent}var {outs[1][1]} = {prefix}a1i - {prefix}b1r")
        e.add(f"{indent}var {outs[3][0]} = {prefix}a1r - {prefix}b1i")
        e.add(f"{indent}var {outs[3][1]} = {prefix}a1i + {prefix}b1r")
    return outs


def _emit_radix5_named(
    e: LineEmitter,
    *,
    indent: str,
    ins: list[tuple[str, str]],
    prefix: str,
    inverse: bool,
) -> list[tuple[str, str]]:
    """Five-point fixed butterfly, algebraically matching rocFFT radix_5.h."""
    (r0, i0), (r1, i1), (r2, i2), (r3, i3), (r4, i4) = ins
    qa = _f32(cos(2.0 * pi / 5.0))          # 0.309016994...
    qb = _f32(sin(2.0 * pi / 5.0))          # 0.951056516...
    qc = _f32(0.5)
    qd = _f32(sin(pi / 5.0))                # 0.587785252...
    sgn = -1 if inverse else 1

    # Common sums/differences used by the rocFFT form.
    e.add(f"{indent}var {prefix}s14r = {r1} + {r4}")
    e.add(f"{indent}var {prefix}s14i = {i1} + {i4}")
    e.add(f"{indent}var {prefix}s23r = {r2} + {r3}")
    e.add(f"{indent}var {prefix}s23i = {i2} + {i3}")
    e.add(f"{indent}var {prefix}d14r = {r1} - {r4}")
    e.add(f"{indent}var {prefix}d14i = {i1} - {i4}")
    e.add(f"{indent}var {prefix}d23r = {r2} - {r3}")
    e.add(f"{indent}var {prefix}d23i = {i2} - {i3}")

    outs = [_out(prefix, k) for k in range(5)]
    e.add(f"{indent}var {outs[0][0]} = {r0} + {prefix}s14r + {prefix}s23r")
    e.add(f"{indent}var {outs[0][1]} = {i0} + {prefix}s14i + {prefix}s23i")

    # These bases are the real-only cosine combinations.  The sine terms are
    # then added/subtracted in conjugate pairs, which cuts duplicated work.
    e.add(
        f"{indent}var {prefix}b1r = {r0} - {prefix}s23r * {qc} "
        f"+ (({r1} - {r2}) + ({r4} - {r3})) * {qa}"
    )
    e.add(
        f"{indent}var {prefix}b1i = {i0} - {prefix}s23i * {qc} "
        f"+ (({i1} - {i2}) + ({i4} - {i3})) * {qa}"
    )
    e.add(
        f"{indent}var {prefix}b2r = {r0} - {prefix}s14r * {qc} "
        f"+ (({r2} - {r1}) + ({r3} - {r4})) * {qa}"
    )
    e.add(
        f"{indent}var {prefix}b2i = {i0} - {prefix}s14i * {qc} "
        f"+ (({i2} - {i1}) + ({i3} - {i4})) * {qa}"
    )
    e.add(f"{indent}var {prefix}c1r = {prefix}d14i * {qb} + {prefix}d23i * {qd}")
    e.add(f"{indent}var {prefix}c1i = {prefix}d14r * {qb} + {prefix}d23r * {qd}")
    e.add(f"{indent}var {prefix}c2r = -{prefix}d23i * {qb} + {prefix}d14i * {qd}")
    e.add(f"{indent}var {prefix}c2i = -{prefix}d23r * {qb} + {prefix}d14r * {qd}")

    if sgn > 0:  # forward
        e.add(f"{indent}var {outs[1][0]} = {prefix}b1r + {prefix}c1r")
        e.add(f"{indent}var {outs[1][1]} = {prefix}b1i - {prefix}c1i")
        e.add(f"{indent}var {outs[4][0]} = {prefix}b1r - {prefix}c1r")
        e.add(f"{indent}var {outs[4][1]} = {prefix}b1i + {prefix}c1i")
        e.add(f"{indent}var {outs[2][0]} = {prefix}b2r + {prefix}c2r")
        e.add(f"{indent}var {outs[2][1]} = {prefix}b2i - {prefix}c2i")
        e.add(f"{indent}var {outs[3][0]} = {prefix}b2r - {prefix}c2r")
        e.add(f"{indent}var {outs[3][1]} = {prefix}b2i + {prefix}c2i")
    else:
        e.add(f"{indent}var {outs[1][0]} = {prefix}b1r - {prefix}c1r")
        e.add(f"{indent}var {outs[1][1]} = {prefix}b1i + {prefix}c1i")
        e.add(f"{indent}var {outs[4][0]} = {prefix}b1r + {prefix}c1r")
        e.add(f"{indent}var {outs[4][1]} = {prefix}b1i - {prefix}c1i")
        e.add(f"{indent}var {outs[2][0]} = {prefix}b2r - {prefix}c2r")
        e.add(f"{indent}var {outs[2][1]} = {prefix}b2i + {prefix}c2i")
        e.add(f"{indent}var {outs[3][0]} = {prefix}b2r + {prefix}c2r")
        e.add(f"{indent}var {outs[3][1]} = {prefix}b2i - {prefix}c2i")
    return outs


def _emit_small_named(
    e: LineEmitter,
    *,
    indent: str,
    ins: list[tuple[str, str]],
    prefix: str,
    radix: int,
    inverse: bool,
) -> list[tuple[str, str]]:
    if radix == 2:
        return _emit_radix2_named(e, indent=indent, ins=ins, prefix=prefix)
    if radix == 3:
        return _emit_radix3_named(
            e, indent=indent, ins=ins, prefix=prefix, inverse=inverse
        )
    if radix == 4:
        return _emit_radix4_named(
            e, indent=indent, ins=ins, prefix=prefix, inverse=inverse
        )
    if radix == 5:
        return _emit_radix5_named(
            e, indent=indent, ins=ins, prefix=prefix, inverse=inverse
        )
    raise ValueError(f"no small fixed butterfly for radix {radix}")


def _emit_factorized(
    e: LineEmitter,
    *,
    indent: str,
    radix: int,
    a: int,
    b: int,
    inverse: bool,
    on_output: OutputCallback | None = None,
) -> None:
    """Emit Cooley-Tukey N=a*b using fixed small butterflies.

    Input is split as n=n1+a*n2.  First perform b-point FFTs over n2,
    multiply by W_N^(n1*k2), then perform a-point FFTs over n1.  Natural
    output is k=k2+b*k1.
    """
    if a * b != radix:
        raise ValueError("invalid factorization")

    groups: list[list[tuple[str, str]]] = []
    for n1 in range(a):
        ins = [(f"rr{n1 + a*n2}", f"ii{n1 + a*n2}") for n2 in range(b)]
        groups.append(
            _emit_small_named(
                e,
                indent=indent,
                ins=ins,
                prefix=f"ctg{n1}_",
                radix=b,
                inverse=inverse,
            )
        )

    for k2 in range(b):
        across: list[tuple[str, str]] = []
        for n1 in range(a):
            sr, si = groups[n1][k2]
            if n1 == 0 or k2 == 0:
                across.append((sr, si))
            else:
                tr = f"cttw_{k2}_{n1}r"
                ti = f"cttw_{k2}_{n1}i"
                _emit_complex_twiddle(
                    e,
                    indent=indent,
                    src_r=sr,
                    src_i=si,
                    dst_r=tr,
                    dst_i=ti,
                    exponent=n1 * k2,
                    modulus=radix,
                    inverse=inverse,
                )
                across.append((tr, ti))

        final = _emit_small_named(
            e,
            indent=indent,
            ins=across,
            prefix=f"ctf{k2}_",
            radix=a,
            inverse=inverse,
        )
        for k1, (fr, fi) in enumerate(final):
            k = k2 + b * k1
            e.add(f"{indent}var or{k} = {fr}")
            e.add(f"{indent}var oi{k} = {fi}")
            if on_output is not None:
                on_output(k)


def _emit_symmetric_odd_radix(
    e: LineEmitter,
    *,
    indent: str,
    radix: int,
    inverse: bool,
    on_output: OutputCallback | None = None,
) -> None:
    """Symmetry-reduced fixed DFT for odd prime/small odd radices.

    It forms x[j]+x[N-j] and x[j]-x[N-j] once, then reuses those values for
    the conjugate output pair k and N-k.  This is the same structural idea
    visible in rocFFT's fixed radix-11/13/17 butterflies, while keeping the
    generator compact and independent of rocFFT's named Qxx constants.
    """
    half = (radix - 1) // 2
    sign = 1.0 if inverse else -1.0

    for j in range(1, half + 1):
        q = radix - j
        e.add(f"{indent}var p{j}r = rr{j} + rr{q}")
        e.add(f"{indent}var p{j}i = ii{j} + ii{q}")
        e.add(f"{indent}var m{j}r = rr{j} - rr{q}")
        e.add(f"{indent}var m{j}i = ii{j} - ii{q}")

    e.add(f"{indent}var or0 = rr0")
    e.add(f"{indent}var oi0 = ii0")
    for j in range(1, half + 1):
        e.add(f"{indent}or0 += p{j}r")
        e.add(f"{indent}oi0 += p{j}i")
    if on_output is not None:
        on_output(0)

    for k in range(1, half + 1):
        nk = radix - k
        e.add(f"{indent}var or{k} = rr0")
        e.add(f"{indent}var oi{k} = ii0")
        e.add(f"{indent}var or{nk} = rr0")
        e.add(f"{indent}var oi{nk} = ii0")
        for j in range(1, half + 1):
            theta = 2.0 * pi * k * j / radix
            c = _f32(cos(theta))
            s = _f32(sin(theta))
            # Forward uses exp(-j theta): +sin*m_i in real, -sin*m_r in imag.
            if sign < 0:
                e.add(f"{indent}or{k} += p{j}r * {c} + m{j}i * {s}")
                e.add(f"{indent}oi{k} += p{j}i * {c} - m{j}r * {s}")
                e.add(f"{indent}or{nk} += p{j}r * {c} - m{j}i * {s}")
                e.add(f"{indent}oi{nk} += p{j}i * {c} + m{j}r * {s}")
            else:
                e.add(f"{indent}or{k} += p{j}r * {c} - m{j}i * {s}")
                e.add(f"{indent}oi{k} += p{j}i * {c} + m{j}r * {s}")
                e.add(f"{indent}or{nk} += p{j}r * {c} + m{j}i * {s}")
                e.add(f"{indent}oi{nk} += p{j}i * {c} - m{j}r * {s}")
        if on_output is not None:
            on_output(k)
            on_output(nk)


def emit_butterfly(
    e: LineEmitter,
    *,
    indent: str,
    radix: int,
    inverse: bool,
    on_output: OutputCallback | None = None,
) -> None:
    """Emit one supported fixed-radix butterfly.

    Unsupported radices are rejected. There is intentionally no fallback path.

    `on_output`, if given, is called with each output index k as soon as
    or{k}/oi{k} are fully computed -- see the OutputCallback docstring.
    """
    if radix not in SUPPORTED_RADICES:
        supported = ", ".join(str(r) for r in sorted(SUPPORTED_RADICES))
        raise ValueError(
            f"radix-{radix} is not supported; supported radices are {{{supported}}}"
        )

    e.add(f"{indent}# fixed radix-{radix} butterfly")

    if radix in (2, 3, 4, 5):
        ins = [(f"rr{k}", f"ii{k}") for k in range(radix)]
        outs = _emit_small_named(
            e,
            indent=indent,
            ins=ins,
            prefix="o",
            radix=radix,
            inverse=inverse,
        )
        assert outs == [(f"or{k}", f"oi{k}") for k in range(radix)]
        if on_output is not None:
            for k in range(radix):
                on_output(k)
        return

    if radix == 6:
        _emit_factorized(
            e, indent=indent, radix=6, a=2, b=3, inverse=inverse, on_output=on_output
        )
        return
    if radix == 8:
        _emit_factorized(
            e, indent=indent, radix=8, a=2, b=4, inverse=inverse, on_output=on_output
        )
        return
    if radix == 9:
        _emit_factorized(
            e, indent=indent, radix=9, a=3, b=3, inverse=inverse, on_output=on_output
        )
        return
    if radix == 10:
        _emit_factorized(
            e, indent=indent, radix=10, a=2, b=5, inverse=inverse, on_output=on_output
        )
        return
    if radix == 16:
        _emit_factorized(
            e, indent=indent, radix=16, a=4, b=4, inverse=inverse, on_output=on_output
        )
        return

    if radix in (7, 11, 13, 17):
        _emit_symmetric_odd_radix(
            e,
            indent=indent,
            radix=radix,
            inverse=inverse,
            on_output=on_output,
        )
        return

    # Kept as a defensive invariant in case SUPPORTED_RADICES and this
    # dispatcher are edited inconsistently in the future.
    raise AssertionError(f"missing emitter branch for supported radix-{radix}")