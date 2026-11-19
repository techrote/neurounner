"""H62 frozen-import compatibility shim: exact H51 float64 receiver margins.

No fitting, labels or holdout access. This adapter preserves H62's preregistered
source unchanged and delegates to the earlier independently tested H51 analytic
209-parameter ReLU route-margin formula. Its source is pinned prior to H62
prospective inference. This is NOT H61's historical LP training implementation.
"""
from h51_interval_invariant_audit import margins as fast_margin
