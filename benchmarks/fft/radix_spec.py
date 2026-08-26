from __future__ import annotations

"""The one fact both planning/ and codegen/ need about which radices this
generator can turn into an actual butterfly: SUPPORTED_RADICES. Lives here,
outside both packages, because it isn't a codegen decision (planning uses it
to factor N and validate stage radices, independent of how a butterfly gets
rendered) or a planning decision (codegen uses the same set to validate a
stage it's about to emit) -- it's a fact about this generator's own coverage
that both sides must agree on, so neither package should be the other's
source for it.
"""

SUPPORTED_RADICES = frozenset((2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17))
