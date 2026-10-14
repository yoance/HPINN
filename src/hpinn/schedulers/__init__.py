"""Combined Learning Rate

Learning rate scheduler that combines exponential decay
with piecewise constant decay.
"""

from .combined_decay import CombinedDecay

__all__ = [
    "CombinedDecay",
]
