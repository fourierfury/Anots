"""Selection masks and soft tapering (design §12)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def rectangle_mask(
    n_freq: int,
    n_frames: int,
    freq_bins: tuple[int, int],
    frame_bins: tuple[int, int],
) -> np.ndarray:
    """Binary mask with a rectangular region set to 1 (design §6.2)."""
    mask = np.zeros((n_freq, n_frames), dtype=np.float64)
    f0, f1 = freq_bins
    t0, t1 = frame_bins
    mask[f0:f1, t0:t1] = 1.0
    return mask


def feather(mask: np.ndarray, taper_bins: int = 8) -> np.ndarray:
    """Hann-ramp the mask interior near its boundary; hard edges ring on ISTFT (§12)."""
    binary = mask > 0
    if taper_bins <= 0 or not binary.any():
        return mask.astype(np.float64)

    dist = distance_transform_edt(binary)
    ramp = np.clip(dist / taper_bins, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * ramp)
