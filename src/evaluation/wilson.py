"""Wilson score interval for binomial proportions (paper reporting)."""

from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return (lo, hi) on the probability scale for successes/total."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_pct_str(successes: int, total: int, z: float = 1.96) -> str:
    """Format as ``[lo, hi]`` in percent with one decimal place."""
    lo, hi = wilson_interval(successes, total, z=z)
    return f"[{100.0 * lo:.1f}, {100.0 * hi:.1f}]"
