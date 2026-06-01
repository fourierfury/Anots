"""The rendering half of the Transform/renderer split.

All matplotlib/axis concerns live here; the ``Transform`` core stays display-free.
The renderer asks the transform only for domain-agnostic facts (axis label, the y
value at a row), so a new view works without touching this module.
"""

from __future__ import annotations

import numpy as np

from ..transforms.base import Transform


class SpectrogramRenderer:
    """Configures a matplotlib axis to display a transform's coefficient matrix."""

    def __init__(self, transform: Transform):
        self.transform = transform

    def extent(self, coeffs: np.ndarray, sr: int, duration_s: float) -> tuple[float, float, float, float]:
        """imshow ``extent`` (x: time, y: the transform's display-y range)."""
        y0, y1 = self.transform.y_extent(coeffs, sr)
        return (0.0, duration_s, y0, y1)

    def draw(self, ax, coeffs: np.ndarray, sr: int, duration_s: float, cmap: str = "viridis") -> None:
        ax.imshow(
            self.transform.display_db(coeffs),
            origin="lower", aspect="auto", cmap=cmap,
            extent=self.extent(coeffs, sr, duration_s),
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.transform.y_label)
        ticks = self.transform.y_ticks(coeffs, sr)
        if ticks is not None:
            ax.set_yticks(ticks[0])
            ax.set_yticklabels(ticks[1])
