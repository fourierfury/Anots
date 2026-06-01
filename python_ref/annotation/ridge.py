"""Magnetic ridge follower: least-cost path between two points, plus its tube mask
(design §6.5).

The user clicks a start and an end bin; the algorithm treats the spectrogram as a
weighted DAG (one node per time-frequency bin, edges between adjacent frames with a
bounded frequency jump) and finds the path of maximum energy between them by dynamic
programming. The 1-D path is then widened into a soft-edged "tube" selection mask.

Like :mod:`overlap`, this is transform-free: it takes a plain magnitude matrix, so it
follows ridges in any view (STFT, CQT, scalogram) without modification. Complexity is
``O(T * F * K)`` (design §6.5).
"""

from __future__ import annotations

import numpy as np

# Design §6.5 defaults.
MAX_JUMP = 20      # K: max frequency-bin jump between adjacent frames
SMOOTHNESS = 0.15  # lambda: penalty weight on frequency jumps
TUBE_WIDTH = 8     # W: half-width of the soft selection tube, in bins


def ridge_path(
    energy: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    k: int = MAX_JUMP,
    lam: float = SMOOTHNESS,
) -> list[tuple[int, int]]:
    """Least-cost path of ``(frame, freq_bin)`` pairs from ``start`` to ``end``.

    ``energy`` is a non-negative magnitude matrix ``(n_freq, n_frames)``; it is
    normalized to ``[0, 1]`` internally. ``start``/``end`` are ``(frame, freq_bin)``.
    The per-bin cost is ``(1 - normalized_energy)`` so the path is drawn toward bright
    bins; ``lam * |df| / k`` penalizes large jumps, yielding smooth paths (design §6.5).
    Endpoints may be given in either time order; the returned path always runs from the
    given ``start`` to the given ``end``.
    """
    n_freq, n_frames = energy.shape
    (t0, f0), (t1, f1) = start, end
    f0 = int(np.clip(f0, 0, n_freq - 1))
    f1 = int(np.clip(f1, 0, n_freq - 1))

    flip = t1 < t0
    if flip:
        (t0, f0), (t1, f1) = (t1, f1), (t0, f0)
    if t1 == t0:
        return [(t0, f0)]

    e = np.asarray(energy, dtype=np.float64)
    peak = e.max()
    node = 1.0 - (e / peak if peak > 0 else e)  # cost to occupy a bin

    dp = np.full(n_freq, np.inf)
    dp[f0] = node[f0, t0]
    back = np.full((t1 - t0 + 1, n_freq), -1, dtype=int)  # back[step][f] = prev freq

    for step, t in enumerate(range(t0 + 1, t1 + 1), start=1):
        best = np.full(n_freq, np.inf)
        arg = np.full(n_freq, -1, dtype=int)
        for d in range(-k, k + 1):  # d = f_target - f_prev
            shifted = np.full(n_freq, np.inf)
            if d > 0:
                shifted[d:] = dp[: n_freq - d]
            elif d < 0:
                shifted[: n_freq + d] = dp[-d:]
            else:
                shifted = dp.copy()
            cand = shifted + lam * abs(d) / k
            improve = cand < best
            best[improve] = cand[improve]
            arg[improve] = np.arange(n_freq)[improve] - d  # source freq = f - d
        dp = node[:, t] + best
        back[step] = arg

    # Backtrack from the required endpoint.
    path: list[tuple[int, int]] = []
    f = f1
    for step in range(t1 - t0, -1, -1):
        path.append((t0 + step, f))
        if step > 0:
            f = int(back[step][f])
    path.reverse()  # path currently runs t1 -> t0; restore t0 -> t1
    return path[::-1] if flip else path


def tube_mask(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    width: int = TUBE_WIDTH,
) -> np.ndarray:
    """Soft selection tube of half-width ``width`` bins around ``path`` (design §6.5).

    Each frame on the path is given a ``cos^2`` falloff in frequency: weight 1 on the
    path, smoothly to 0 at ``width`` bins away. The taper suppresses the ISTFT ringing a
    hard edge would cause (design §12). ``shape`` is ``(n_freq, n_frames)``.
    """
    n_freq, n_frames = shape
    mask = np.zeros(shape, dtype=np.float64)
    freqs = np.arange(n_freq)
    for t, fc in path:
        if width <= 0:
            mask[int(fc), t] = 1.0
            continue
        dist = np.abs(freqs - fc)
        within = dist <= width
        mask[within, t] = np.cos(np.pi * dist[within] / (2 * width)) ** 2
    return mask
