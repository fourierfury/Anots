"""Scalogram view via the undecimated (stationary) wavelet transform (design §4.3.6).

**Reconstruction is genuinely exact here (~1e-13), fulfilling design §11's "inverse
wavelet, exact" — unlike CQT/chroma, which had to bridge through the STFT.** The
reason is a measured one: the *continuous* wavelet transform has no exact discrete
inverse (ssqueezepy's ``icwt`` floors at ~1e-3, never the 1e-5 reference tolerance —
the same lesson ``icqt`` taught us), but the *undecimated* wavelet transform is a
perfect-reconstruction filter bank.

The engine is PyWavelets' SWT-based multiresolution analysis (``pywt.mra``), which
returns one **additive** real band-signal per dyadic scale (plus a low-pass residual);
the bands sum back to the signal to ~1e-15. A selection is therefore reconstructed by
summing the selected ``band x time`` content — exact by linearity (positive + negative
== original, validated like CQT/chroma). Because the transform is undecimated, every
band keeps full time resolution, so the scalogram's reason to exist — transient/onset
localization (design §13: gunshots, sharp onsets) — survives into the reconstructed
audio, not just the picture.

Cost of honesty: the scale axis is **dyadic** (one row per octave-ish band), coarser in
frequency than a continuous-CWT scalogram. Time precision, the point of this view, is
full. A fine continuous-CWT *display* overlay is a possible future native nicety; it is
deliberately not in the reconstruction path.

PyWavelets is used as a reference-grade library (design skill 14), never on the native
audio thread; the native core reimplements the SWT filter bank directly.
"""

from __future__ import annotations

import numpy as np
import pywt

from ..annotation.model import ExtractionMode
from ..annotation.reconstruct import effective_mask
from ..dsp import amplitude_to_db
from ..params import SCALOGRAM, ScalogramParams
from .base import Transform


class ScalogramTransform(Transform):
    name = "scalogram"
    accuracy_class = "exact"  # undecimated wavelet recon is perfect (~1e-13); see module docstring
    y_label = "Scale (wavelet pseudo-frequency)"

    def __init__(self, params: ScalogramParams = SCALOGRAM):
        self.params = params
        self._cf = pywt.central_frequency(params.wavelet)  # cycles/sample of the mother wavelet

    # --- engine: additive multiresolution bands (low freq -> high freq) ---

    def _effective_level(self, n: int) -> int:
        """Clamp the requested level to what the (padded) length can support."""
        return max(1, min(self.params.level, pywt.swt_max_level(self._padded_len(n))))

    def _padded_len(self, n: int) -> int:
        """Next length that is a multiple of ``2**level`` (SWT requires it)."""
        block = 1 << self.params.level
        return ((n + block - 1) // block) * block

    def _bands(self, y: np.ndarray) -> np.ndarray:
        """``(n_bands, len(y))`` additive real bands, ordered low -> high frequency.

        Row 0 is the low-pass residual; rows 1..L are detail bands from coarsest to
        finest. The rows sum to ``y`` (to ~1e-15), which is what makes reconstruction
        exact. The signal is reflect-padded to a valid SWT length, then trimmed back.
        """
        n = len(y)
        level = self._effective_level(n)
        block = 1 << level
        m = ((n + block - 1) // block) * block
        yp = np.pad(np.asarray(y, dtype=np.float64), (0, m - n), mode="reflect") if m > n else np.asarray(y, dtype=np.float64)
        comps = pywt.mra(yp, self.params.wavelet, level=level, transform="swt")
        return np.asarray(comps, dtype=np.float64)[:, :n]

    def forward(self, y: np.ndarray, sr: int) -> np.ndarray:
        return self._bands(y)

    def display_db(self, coeffs: np.ndarray) -> np.ndarray:
        return amplitude_to_db(np.abs(coeffs))

    def reconstruct(
        self,
        y: np.ndarray,
        coeffs: np.ndarray,
        mask: np.ndarray,
        mode: ExtractionMode | str = ExtractionMode.POSITIVE,
        sr: int | None = None,
        length: int | None = None,
    ) -> np.ndarray:
        # The bands ARE the (real, additive) coefficients, so summing the selected
        # band x time content reconstructs directly — no separate inverse needed.
        clip = (coeffs * effective_mask(mask, mode)).sum(axis=0)
        if length is not None and length != clip.shape[0]:
            out = np.zeros(length, dtype=np.float64)
            k = min(length, clip.shape[0])
            out[:k] = clip[:k]
            return out
        return clip

    # --- coordinate mapping: dyadic pseudo-frequency <-> band row ---

    def _band_freqs(self, n_bands: int, sr: int) -> np.ndarray:
        """Pseudo-frequency (Hz) at each band row, low row -> high row."""
        level = n_bands - 1
        freqs = np.empty(n_bands)
        # Row 0 = low-pass residual: one octave below the coarsest detail band.
        freqs[0] = self._cf * sr / 2 ** (level + 1)
        for row in range(1, n_bands):          # row r -> detail level (level + 1 - r)
            freqs[row] = self._cf * sr / 2 ** (level + 1 - row)
        return freqs

    def time_to_col(self, t: float, sr: int) -> int:
        return round(t * sr)

    def col_to_time(self, col: int, sr: int) -> float:
        return col / sr

    def value_to_row(self, hz: float, sr: int) -> int:
        n_bands = self.params.level + 1
        freqs = self._band_freqs(n_bands, sr)
        if hz <= 0:
            return 0
        return int(np.argmin(np.abs(np.log2(freqs) - np.log2(hz))))  # nearest in log-frequency

    def row_to_value(self, row: int, sr: int) -> float:
        n_bands = self.params.level + 1
        row = int(np.clip(row, 0, n_bands - 1))
        return float(self._band_freqs(n_bands, sr)[row])

    # Bands are log-spaced scales, so the display y-axis is band index with Hz labels.
    def row_to_display_y(self, row: int, sr: int) -> float:
        return float(row)

    def y_extent(self, coeffs: np.ndarray, sr: int) -> tuple[float, float]:
        return (0.0, float(coeffs.shape[0]))

    def y_ticks(self, coeffs: np.ndarray, sr: int) -> tuple[list, list]:
        n_bands = coeffs.shape[0]
        positions = list(range(n_bands))
        labels = [f"{f:.0f} Hz" for f in self._band_freqs(n_bands, sr)]
        return (positions, labels)

    def provenance(self) -> dict:
        # Reconstruction is wavelet-native — there is no fft_size/hop/window here, so
        # these keys land in the annotation's transform_params (the honest record of how
        # the audio was produced). See progress.md: the Annotation model still defaults
        # its fft_* fields, a wart to clean up when it gains per-domain recon provenance.
        return {
            "scalogram_wavelet": self.params.wavelet,
            "scalogram_level": self.params.level,
        }
