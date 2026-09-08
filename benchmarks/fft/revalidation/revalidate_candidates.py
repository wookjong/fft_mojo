from __future__ import annotations

"""The deterministic candidate list `revalidate_cost_model.py` runs --
kept as its own module so the *dataset composition* is reviewable and
editable independent of the harness plumbing. See that module's own
docstring for why this replaces the lost 55-candidate/14-N dataset with a
hand-specified list instead of a search-generated one.

Covers, per docs/active_ndp_units_cost_task.md's own Phase B/E asks:
- matched pairs (same split/radix/tile, only leaf strategy differs) at a
  small single-leaf N (216), the mixed-leaf N this whole investigation
  centers on (960), and one large N (2048) that also needs a real split.
- a representative-N set for ranking regression spanning small single-leaf
  (216, 144), mixed-leaf/transpose (960), a radix-5-genuine-middle-stage
  case with known spill behavior (630), an asymmetric near/far split
  (512), a second large split (1024), and one very large split (2048).
"""

from revalidate_cost_model import CandidateSpec

_COOP = lambda w: {"cooperative_workers": w}
_PERSISTENT = {"persistent_leaf": True}
_NONCOOP: dict = {}

CANDIDATES: list[CandidateSpec] = [
    # ---- N=216: small, single fused leaf (no split needed) ----
    CandidateSpec(216, "noncoop", _NONCOOP),
    CandidateSpec(216, "coop4", _COOP(4)),
    CandidateSpec(216, "persistent", _PERSISTENT),

    # ---- N=144: second small single-leaf point, different radix mix ----
    CandidateSpec(144, "noncoop", _NONCOOP),
    CandidateSpec(144, "coop4", _COOP(4)),
    CandidateSpec(144, "persistent", _PERSISTENT),

    # ---- N=512: asymmetric near/far split ----
    CandidateSpec(512, "noncoop", _NONCOOP),
    CandidateSpec(512, "coop4", _COOP(4)),
    CandidateSpec(512, "persistent", _PERSISTENT),

    # ---- N=630: transpose-containing, radix-5-genuine-middle spill case ----
    CandidateSpec(630, "noncoop", _NONCOOP),
    CandidateSpec(630, "coop4", _COOP(4)),
    CandidateSpec(630, "persistent", _PERSISTENT),

    # ---- N=960: the mixed-leaf/transpose case this investigation centers
    # on -- full worker sweep, matches Phase 1.5/2/6's own existing data
    # for a direct cross-check against this new harness. ----
    CandidateSpec(960, "noncoop", _NONCOOP),
    CandidateSpec(960, "coop2", _COOP(2)),
    CandidateSpec(960, "coop4", _COOP(4)),
    CandidateSpec(960, "coop8", _COOP(8)),
    CandidateSpec(960, "persistent", _PERSISTENT),
    CandidateSpec(960, "mixed_persistent_near_noncoop_far", {
        "forced_worker_sequence": ("persistent", None),
    }),

    # ---- N=1024: second large split ----
    CandidateSpec(1024, "noncoop", _NONCOOP),
    CandidateSpec(1024, "coop4", _COOP(4)),
    CandidateSpec(1024, "persistent", _PERSISTENT),

    # ---- N=2048: large-N matched pair (Phase B's "large N >= 1") ----
    CandidateSpec(2048, "noncoop", _NONCOOP),
    CandidateSpec(2048, "coop4", _COOP(4)),
    CandidateSpec(2048, "persistent", _PERSISTENT),
]
