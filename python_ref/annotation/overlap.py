"""Pairwise overlap detection between annotation layers (design §5.4).

After any layer change, the GUI runs this to surface two situations:

* **Share** (IoU > 0.30): two layers cover substantially the same time-frequency
  region. Often intentional (overlapping simultaneous sources), so it is an
  informational yellow flag — never blocking.
* **Duplicate** (IoU > 0.95): two layers are near-identical. With the *same* label
  this is the one data-quality risk worth flagging prominently — the same acoustic
  content exported twice as the same training example.

This module is deliberately transform-free: IoU is a property of the masks alone, so
it stays on the Qt-free side of the boundary and applies to every view unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Design §5.4 thresholds.
SHARE_THRESHOLD = 0.30
DUPLICATE_THRESHOLD = 0.95


class OverlapLevel(str, Enum):
    NONE = "none"            # IoU <= 0.30: no indicator
    SHARE = "share"          # IoU > 0.30: yellow — layers share significant area
    DUPLICATE = "duplicate"  # IoU > 0.95: orange — near-identical footprint


def iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.0) -> float:
    """Intersection over Union of two selection masks (design §5.4).

    Masks are feathered (float in [0, 1]); a bin counts as selected where its value
    exceeds ``threshold`` (default: any nonzero membership), matching the set
    semantics ``|A AND B| / |A OR B|`` in the spec. Returns 0.0 for an empty union.
    """
    A = a > threshold
    B = b > threshold
    union = int(np.count_nonzero(A | B))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(A & B)) / union


def classify(value: float) -> OverlapLevel:
    """Map an IoU value to its warning level (design §5.4 thresholds)."""
    if value > DUPLICATE_THRESHOLD:
        return OverlapLevel.DUPLICATE
    if value > SHARE_THRESHOLD:
        return OverlapLevel.SHARE
    return OverlapLevel.NONE


@dataclass
class OverlapWarning:
    """One overlapping layer pair, identified by position in the session's layer list."""

    i: int
    j: int
    iou: float
    level: OverlapLevel
    same_label: bool

    @property
    def is_duplicate_risk(self) -> bool:
        """Same label + near-identical footprint: the same content labeled twice —
        the data-quality risk design §5.4 says to flag prominently."""
        return self.same_label and self.level is OverlapLevel.DUPLICATE


def detect_overlaps(layers, threshold: float = 0.0) -> list[OverlapWarning]:
    """Pairwise IoU over every layer pair, in O(n^2) (design §5.4).

    Returns only pairs that actually overlap (IoU > 0), each tagged with its warning
    level; the caller shows an indicator for ``SHARE``/``DUPLICATE`` and ignores
    ``NONE``. Order is stable: pairs are yielded as ``(i, j)`` with ``i < j``.
    """
    warnings: list[OverlapWarning] = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            value = iou(layers[i].mask, layers[j].mask, threshold)
            if value <= 0.0:
                continue
            warnings.append(OverlapWarning(
                i=i, j=j, iou=value,
                level=classify(value),
                same_label=layers[i].label == layers[j].label,
            ))
    return warnings
