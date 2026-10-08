"""FP32 direct Stockham register/radix decisions at VkFFT 066a17c17068.

Source: VkFFTGetRegistersPerThread (ordinary precision, primes 2/3/5/7),
the scheduler's non-Rader thread-cap scaling, VkFFTOptimizeRadixKernels
(registerBoost=1), MinMaxRegisterCheck and first-stage register swap.
No M2NDP heuristics or butterfly vocabulary participate in these choices.
"""
from dataclasses import dataclass


def fp32_base_registers(count2: int, count3: int, count5: int, count7: int) -> dict[int, int]:
    """Ordinary precision branch with loc_multipliers[11]=[13]=0.

    Pure power-of-two scheduling is handled by the caller; its register
    grouping depends on FFT length and batch rather than a presence table.
    """
    if any(type(c) is not int or c < 0 for c in (count2, count3, count5, count7)):
        raise ValueError("Prime multiplicities must be nonnegative integers")
    r = {2: 0, 3: 0, 5: 0, 7: 0}
    if count2:
        if count3:
            if count5:
                if count7:
                    r[2] = 14 if count2 == 1 else (12 if count2 <= 3 else 16)
                    r[3], r[5], r[7] = (15 if count2 == 1 else 12), 15, 14
                else:
                    r[2] = r[3] = 6 if count2 == 1 else 12
                    r[5] = 5 if count2 == 1 else 10
            elif count7:
                r[2], r[3], r[7] = (6 if count2 <= 2 else 8), 6, 7
            else:
                r[2] = r[3] = 6 if count2 == 1 or count2 == count3 else 12
        elif count5:
            r[2], r[5], r[7] = (10 if count2 <= 2 else 8), 10, (7 if count7 else 0)
        elif count7:
            r[2], r[7] = (14, 14) if count2 <= 2 else (8, 7)
        else:
            raise ValueError("Pure power-of-two registers require the grouping scheduler")
    elif count3:
        if count5:
            r[3], r[5], r[7] = 15, 15, (14 if count7 else 0)
        elif count7:
            r[3], r[7] = (6 if count3 == 1 else 9), 7
        else:
            r[3] = 3 if count3 == 1 else 9
    elif count5:
        r[5], r[7] = 5, (7 if count7 else 0)
    elif count7:
        r[7] = 7
    else:
        raise ValueError("No direct prime factors")
    return r


def prime_multiplicities(length: int) -> dict[int, int]:
    if type(length) is not int or length < 2:
        raise ValueError("FFT length must be an integer >= 2")
    counts = {p: 0 for p in (2, 3, 5, 7)}
    residual = length
    for prime in counts:
        while residual % prime == 0:
            counts[prime] += 1
            residual //= prime
    if residual != 1:
        raise ValueError(f"Direct register scheduler has residual factor {residual}")
    return counts


def composite_registers(base: dict[int, int]) -> dict[int, int]:
    r = dict.fromkeys(range(33), 0)
    r.update(base)
    for radix in (4, 8, 16, 32):
        r[radix] = r[2] if r[2] % radix == 0 else 0
    for radix, a, b in ((6, 2, 3), (10, 2, 5), (14, 2, 7), (15, 3, 5)):
        r[radix] = min(r[a], r[b])
    if r[2] >= 12 and r[3] >= 12:
        r[12] = min(r[2], r[3])
        if r[12] % 12:
            r[12] = 0
    r[9] = r[3] if r[3] % 9 == 0 else 0
    return r


@dataclass(frozen=True)
class DirectRadixPlan:
    radices: tuple[int, ...]
    min_registers: int
    max_registers: int
    registers_per_radix: tuple[tuple[int, int], ...]


def plan_direct_radices(
    length: int, *, pow2_registers: int | None = None,
    max_batch_coalesced: int = 1, upload_id: int = 0, max_threads: int = 1024,
) -> DirectRadixPlan:
    counts = prime_multiplicities(length)
    base = ({2: pow2_registers, 3: 0, 5: 0, 7: 0} if pow2_registers is not None
            else fp32_base_registers(*(counts[p] for p in (2, 3, 5, 7))))
    registers = composite_registers(base)
    minimum = min(v for v in registers.values() if v)
    maximum = max(registers.values())
    if max_batch_coalesced * length > max_threads * minimum:
        if upload_id == 0 and max_batch_coalesced > 1:
            max_batch_coalesced = max(1, max_threads * minimum // length)
        scale = max(1, -(-max_batch_coalesced * length // (max_threads * minimum)))
        registers = {radix: value * scale for radix, value in registers.items()}
        minimum = min(v for v in registers.values() if v)
        maximum = max(registers.values())
        # Scheduler's no-Rader reduction after scaling (not a generic rule).
        if counts[3] >= 2 and (maximum // minimum) % 3 == 0:
            maximum //= 3
            registers = {radix: value // 3 if value % 9 == 0 else value
                         for radix, value in registers.items()}

    multiplicities = dict.fromkeys(range(33), 0)
    multiplicities.update(counts)
    # The exact upstream combination order matters: r12 precedes r8, etc.
    for radix, factors in (
        (32, {2: 5}), (16, {2: 4}), (15, {3: 1, 5: 1}),
        (14, {2: 1, 7: 1}), (12, {2: 2, 3: 1}), (10, {2: 1, 5: 1}),
        (9, {3: 2}), (8, {2: 3}), (6, {2: 1, 3: 1}), (4, {2: 2}),
    ):
        power2 = radix & (radix - 1) == 0
        eligible = ((registers[radix] > 0 or (power2 and registers[2] % radix == 0))
                    and registers[radix] % radix == 0)
        if not eligible:
            continue
        count = min(multiplicities[p] // exponent for p, exponent in factors.items())
        if count == 0:
            continue
        multiplicities[radix] = count
        for prime, exponent in factors.items():
            multiplicities[prime] -= count * exponent
        if power2 and registers[2] % radix == 0:
            registers[radix] = registers[2]
    radices = [radix for radix in range(32, 1, -1) for _ in range(multiplicities[radix])]
    minimum = min(registers[r] for r in radices if registers[r])
    maximum = max(registers[r] for r in radices)
    if minimum != maximum:
        first = next(i for i, r in enumerate(radices) if registers[r] == minimum)
        radices[0], radices[first] = radices[first], radices[0]
    return DirectRadixPlan(tuple(radices), minimum, maximum,
                           tuple((r, v) for r, v in registers.items() if v))
