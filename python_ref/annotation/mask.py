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


def polygon_mask(
    n_freq: int,
    n_frames: int,
    verts: list[tuple[float, float]],
) -> np.ndarray:
    """Binary mask of a filled closed polygon (the Freehand Lasso, design §6.4).

    ``verts`` are ``(col, row)`` = ``(frame, freq_bin)`` vertices tracing the outline. The
    interior is filled by even-odd ray casting, vectorised over the polygon's bounding box.
    Fewer than 3 vertices yields an empty mask.
    """
    mask = np.zeros((n_freq, n_frames), dtype=np.float64)
    if len(verts) < 3:
        return mask
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    c0, c1 = max(0, int(np.floor(min(xs)))), min(n_frames, int(np.ceil(max(xs))) + 1)
    r0, r1 = max(0, int(np.floor(min(ys)))), min(n_freq, int(np.ceil(max(ys))) + 1)
    if c0 >= c1 or r0 >= r1:
        return mask

    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    x = cc.ravel().astype(np.float64)
    y = rr.ravel().astype(np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        straddles = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        inside ^= straddles & (x < x_cross)
        j = i
    mask[r0:r1, c0:c1] = inside.reshape(r1 - r0, c1 - c0).astype(np.float64)
    return mask


def feather(mask: np.ndarray, taper_bins: int = 8) -> np.ndarray:
    """Hann-ramp the mask interior near its boundary; hard edges ring on ISTFT (§12)."""
    binary = mask > 0
    if taper_bins <= 0 or not binary.any():
        return mask.astype(np.float64)

    dist = distance_transform_edt(binary)
    ramp = np.clip(dist / taper_bins, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * ramp)
