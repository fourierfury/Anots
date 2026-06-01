"""The rendering half of the Transform/renderer split.

All matplotlib/axis concerns live here; the ``Transform`` core stays display-free.
The renderer asks the transform only for domain-agnostic facts (axis label, the y
value at a row), so a new view works without touching this module.
"""

from __future__ import annotations

import numpy as np

from ..transforms.base import Transform


def energy_bounds(db: np.ndarray, top_db: float = 60.0) -> tuple[int, int, int, int] | None:
    """Inclusive ``(row_lo, row_hi, col_lo, col_hi)`` of bins within ``top_db`` of the peak.

    Used by "fit to content": the frequency/time window where the signal actually lives,
    so the view can zoom to it instead of showing a mostly-empty full-range axis. Returns
    ``None`` for silence (nothing within ``top_db`` of the peak). Works on any view's dB
    matrix — the caller maps rows/cols to display units through the transform.
    """
    peak = float(db.max())
    strong = db > (peak - top_db)
    if not strong.any():
        return None
    rows = np.where(strong.any(axis=1))[0]
    cols = np.where(strong.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def occupied_band(mag: np.ndarray, central: float = 0.99) -> tuple[int, int] | None:
    """Row range ``(lo, hi)`` holding ``central`` of the per-frequency energy.

    This is the **occupied bandwidth** (the FCC 99%-power definition): total per-row energy
    is accumulated from both ends, and the band is cut where each tail reaches
    ``(1-central)/2`` of the total — so thin noise tails and empty high-frequency space are
    excluded. Far more robust than a dB-from-peak threshold for "where does the signal live".
    Returns ``None`` for silence. ``mag`` is a magnitude matrix ``(n_freq, n_frames)``; it
    works in any view (rows are this view's frequency/scale bins).
    """
    power = (np.asarray(mag, dtype=np.float64) ** 2).sum(axis=1)  # energy per frequency row
    total = float(power.sum())
    if total <= 0.0:
        return None
    cum = np.cumsum(power)
    tail = (1.0 - central) / 2.0
    lo = int(np.searchsorted(cum, tail * total))
    hi = int(np.searchsorted(cum, (1.0 - tail) * total))
    n = len(power)
    lo = max(0, min(lo, n - 1))
    hi = max(lo, min(hi, n - 1))
    return lo, hi


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
