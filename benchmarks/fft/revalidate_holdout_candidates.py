from __future__ import annotations

"""Phase 2 holdout dataset (docs/active_ndp_units_cost_task.md's "Unseen-N
holdout validation" task, 2026-09-01) -- N values deliberately DISTINCT
from the calibration set `revalidate_candidates.py` used to derive
`persistent_stage_batch_multiplier`/`persistent_extra_round_multiplier`
(144, 216, 512, 630, 960, 1024, 2048). None of these 15 N were used to fit
either coefficient; this tests generalization, not calibration.

Spans (by real radix decomposition, not by hand-waving):
- tiny/small, single-leaf: 12 (4,3), 20 (4,5), 40 (4,2,5)
- small/medium, single-leaf: 84 (4,3,7 -- odd radix), 100 (4,5,5),
  180 (4,3,3,5), 240 (4,4,3,5)
- medium/large, split (lopsided near/far): 360, 384 (radix-2-heavy far
  leaf), 450, 720, 800
- large, split: 1280, 1536, 3072 (largest -- stress test)
"""

from revalidate_cost_model import CandidateSpec

_COOP4 = {"cooperative_workers": 4}
_PERSISTENT = {"persistent_leaf": True}
_NONCOOP: dict = {}

_HOLDOUT_N = [12, 20, 40, 84, 100, 180, 240, 360, 384, 450, 720, 800, 1280, 1536, 3072]

CANDIDATES: list[CandidateSpec] = [
    spec
    for n in _HOLDOUT_N
    for spec in (
        CandidateSpec(n, "noncoop", _NONCOOP),
        CandidateSpec(n, "coop4", _COOP4),
        CandidateSpec(n, "persistent", _PERSISTENT),
    )
]
