from __future__ import annotations

"""M2NDP-side equivalent of rocFFT's own `solution_map` (library/src/
include/solution_map.h -- see gpu_baseline/rocfft.py's own module
docstring for the source citation and commit snapshot): a raw, on-disk
cache of ONE measured baseline-planning result, keyed by the minimum
information needed to reproduce it.

NON-NEGOTIABLE (section 1B.10 of the task this package was built from):
"Do not use this stored result as a new fitted M2NDP cost model." This
module never computes, fits, or exposes anything derived across multiple
entries (no averaging, no interpolation, no "predict an unseen length
from nearby ones") -- it is a flat key -> single-measured-record store,
nothing more, exactly mirroring rocFFT's own `solution_map` role as a
compiled-config CACHE, not a performance model.

Key fields (mirroring rocFFT's own `ProblemKey`/`FMKey` shape -- research
report section 11): target/device profile (identified by its own
dataclass field values, since this project has exactly one real
`TargetProfile` today but the key must not assume that stays true),
planner ("clfft" | "rocfft" | "rocfft-default" | "vkfft"), FFT length, precision (this
project is FP32-only, but the field is kept for honesty about what is and
isn't varied), direction (forward/inverse), and batch/problem shape.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from planning.core.target_profile import TargetProfile

from .common import BaselineStatus


def _target_key(target: TargetProfile) -> str:
    """A target profile's own identity for keying purposes -- every field
    that could change which baseline configuration is legal (scratchpad
    capacity, interleave chunk, register limits, ...), not a hash of the
    whole object (so a key stays readable in the on-disk JSON)."""
    fields = asdict(target)
    return "_".join(f"{k}={fields[k]}" for k in sorted(fields))


@dataclass(frozen=True)
class SolutionKey:
    planner: str  # "clfft" | "rocfft" | "vkfft"
    length: int
    precision: str  # always "fp32" today -- see module docstring
    inverse: bool
    batch: int
    target_key: str

    def as_str(self) -> str:
        return (
            f"{self.planner}|len={self.length}|prec={self.precision}|"
            f"inverse={self.inverse}|batch={self.batch}|{self.target_key}"
        )


@dataclass(frozen=True)
class SolutionRecord:
    """The RAW measured result and selected configuration for one
    `SolutionKey` -- see module docstring's non-negotiable rule: this is
    stored and read back verbatim, never fitted into a formula."""

    status: str  # BaselineStatus.value
    radices: tuple[int, ...]
    gpu_extra: dict[str, object]
    ndp_cycles: int | None
    spill_free: bool | None
    diagnostics: str


@dataclass
class SolutionMap:
    """In-memory + on-disk JSON store. `path`: where `save()`/`load()`
    persist -- callers own the load/save lifecycle explicitly (no implicit
    background I/O), mirroring rocFFT's own explicit
    `read_solution_map_data`/`write_solution_map_data` API shape."""

    path: Path
    _entries: dict[str, SolutionRecord] = field(default_factory=dict)

    def key_for(
        self, *, planner: str, length: int, inverse: bool, batch: int,
        target: TargetProfile, precision: str = "fp32",
    ) -> SolutionKey:
        return SolutionKey(
            planner=planner, length=length, precision=precision, inverse=inverse,
            batch=batch, target_key=_target_key(target),
        )

    def get(self, key: SolutionKey) -> SolutionRecord | None:
        return self._entries.get(key.as_str())

    def put(self, key: SolutionKey, record: SolutionRecord) -> None:
        self._entries[key.as_str()] = record

    def record_result(
        self, *, planner: str, length: int, inverse: bool, batch: int,
        target: TargetProfile, status: BaselineStatus, radices: tuple[int, ...],
        gpu_extra: dict[str, object], ndp_cycles: int | None, spill_free: bool | None,
        diagnostics: str, precision: str = "fp32",
    ) -> SolutionKey:
        """Convenience wrapper building both the key and record from a
        `BaselineResult`-shaped set of fields -- the one write path every
        baseline's own CLI integration uses (see make_fft_kernel.py's
        `--planner` wiring), so every stored entry has the same shape
        regardless of which of the three baselines produced it."""
        key = self.key_for(
            planner=planner, length=length, inverse=inverse, batch=batch,
            target=target, precision=precision,
        )
        record = SolutionRecord(
            status=status.value, radices=tuple(radices), gpu_extra=dict(gpu_extra),
            ndp_cycles=ndp_cycles, spill_free=spill_free, diagnostics=diagnostics,
        )
        self.put(key, record)
        return key

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries = {
            k: SolutionRecord(
                status=v["status"], radices=tuple(v["radices"]), gpu_extra=v["gpu_extra"],
                ndp_cycles=v["ndp_cycles"], spill_free=v["spill_free"], diagnostics=v["diagnostics"],
            )
            for k, v in raw.items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            k: {
                "status": v.status, "radices": list(v.radices), "gpu_extra": v.gpu_extra,
                "ndp_cycles": v.ndp_cycles, "spill_free": v.spill_free, "diagnostics": v.diagnostics,
            }
            for k, v in self._entries.items()
        }
        self.path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._entries)
