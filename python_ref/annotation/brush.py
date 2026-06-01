"""Smart Energy Brush: paint time-frequency bins into a selection mask (design §6.6).

A circular brush stamp adds bins to a selection. In **standard** mode every bin under
the cursor is added; in **threshold** mode only bins whose magnitude (dB) clears a
threshold are added, so the brush grabs the loud content and skips the quiet gaps —
"select by energy", the audio analogue of image editing's "select by colour".

Like the other selection primitives this is transform-free and pure: it operates on a
display-dB matrix and a mask, so it works in any view. A GUI drag is just a sequence of
stamps accumulated into one mask.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RADIUS = 8  # brush radius in bins (the GUI maps the scroll wheel to this)


def energy_brush(
    db: np.ndarray,
    center: tuple[int, int],
    radius: int = DEFAULT_RADIUS,
    threshold_db: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Stamp one circular brush of bins into a selection mask (design §6.6).

    ``db`` is the display-dB matrix ``(n_freq, n_frames)`` (``transform.display_db``);
    ``center`` is ``(row, col)`` in bin coordinates; ``radius`` is in bins. Standard mode
    (``threshold_db is None``) adds every bin within ``radius``; threshold mode adds only
    bins with ``db >= threshold_db``. The stamp is **additive**: pass the running ``mask``
    to accumulate a drag; a fresh mask is created when ``mask is None``. The input mask is
    never mutated.
    """
    n_freq, n_frames = db.shape
    r0, c0 = center
    rows = np.arange(n_freq)[:, None]
    cols = np.arange(n_frames)[None, :]
    within = (rows - r0) ** 2 + (cols - c0) ** 2 <= radius ** 2
    if threshold_db is not None:
        within &= db >= threshold_db

    out = np.zeros((n_freq, n_frames), dtype=np.float64) if mask is None else mask.astype(np.float64).copy()
    out[within] = 1.0
    return out
