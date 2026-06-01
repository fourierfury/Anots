"""Constant-Q transform view (design §4.3.4).

**Reconstruction note (divergence from design §11):** §11 specifies "CQT -> inverse
CQT (exact)". In practice librosa's ``icqt`` round-trip is poor (~9-82% rel RMS, never
1e-5), so it cannot serve as exact ground truth. Instead the CQT here is a *log-frequency
selection-and-view surface* over the exact STFT engine: a mask drawn on the CQT is mapped
to STFT bins and inverted with the 1e-5 ISTFT. The reconstructed audio is therefore exact
for the selected band; the cost is that selection granularity is quantised to STFT bins.
A truly invertible CQT (NSGT) is deferred to the native core. §11 should be annotated to
reflect this.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..annotation.model import ExtractionMode
from ..annotation.reconstruct import effective_mask
from ..dsp import amplitude_to_db, istft, stft
from ..params import CQT, STFT, CqtParams, StftParams
from .base import Transform


class CqtTransform(Transform):
    name = "cqt"
    accuracy_class = "exact"  # audio is an exact STFT reconstruction of the selected band
    y_label = "Note (log frequency)"

    def __init__(self, params: CqtParams = CQT, stft_params: StftParams = STFT):
        self.params = params
        self.stft_params = stft_params

    def forward(self, y: np.ndarray, sr: int) -> np.ndarray:
        return librosa.cqt(
            y, sr=sr,
            hop_length=self.params.hop_length,
            fmin=self.params.fmin,
            n_bins=self.params.n_bins,
            bins_per_octave=self.params.bins_per_octave,
        )

    def display_db(self, coeffs: np.ndarray) -> np.ndarray:
        return amplitude_to_db(np.abs(coeffs))

    def frequencies(self) -> np.ndarray:
        return librosa.cqt_frequencies(
            n_bins=self.params.n_bins,
            fmin=self.params.fmin,
            bins_per_octave=self.params.bins_per_octave,
        )

    def reconstruct(
        self,
        y: np.ndarray,
        coeffs: np.ndarray,
        mask: np.ndarray,
        mode: ExtractionMode | str = ExtractionMode.POSITIVE,
        sr: int | None = None,
        length: int | None = None,
    ) -> np.ndarray:
        S = stft(y, self.stft_params)
        stft_mask = self._mask_to_stft(mask, S.shape, sr)
        return istft(S * effective_mask(stft_mask, mode), self.stft_params, length=length)

    def _mask_to_stft(self, cqt_mask: np.ndarray, stft_shape: tuple[int, int], sr: int) -> np.ndarray:
        """Project a CQT-domain mask onto STFT bins (nearest semitone bin per STFT bin).

        CQT rows are log-spaced; each STFT bin is assigned the mask value of the CQT
        bin whose semitone band contains it. STFT bins outside the CQT range get 0.
        Frame axes align because both transforms share ``hop_length``; any off-by-one
        is reconciled by truncating to the common frame count.
        """
        n_bins_stft, n_frames_stft = stft_shape
        bpo, fmin, n_bins = self.params.bins_per_octave, self.params.fmin, self.params.n_bins
        stft_freqs = np.fft.rfftfreq(self.stft_params.n_fft, 1.0 / sr)

        cqt_freqs = self.frequencies()
        lo = cqt_freqs[0] / 2 ** (0.5 / bpo)   # half a semitone below the lowest bin
        hi = cqt_freqs[-1] * 2 ** (0.5 / bpo)  # half a semitone above the highest bin

        with np.errstate(divide="ignore", invalid="ignore"):
            rows = np.round(bpo * np.log2(np.where(stft_freqs > 0, stft_freqs, np.nan) / fmin))
        cover = (stft_freqs >= lo) & (stft_freqs <= hi)
        rows = np.clip(np.nan_to_num(rows, nan=0.0), 0, n_bins - 1).astype(int)

        n = min(n_frames_stft, cqt_mask.shape[1])
        stft_mask = np.zeros((n_bins_stft, n_frames_stft), dtype=np.float64)
        stft_mask[cover, :n] = cqt_mask[rows[cover][:, None], np.arange(n)]
        return stft_mask

    def time_to_col(self, t: float, sr: int) -> int:
        return round(t * sr / self.params.hop_length)

    def col_to_time(self, col: int, sr: int) -> float:
        return col * self.params.hop_length / sr

    def value_to_row(self, hz: float, sr: int) -> int:
        if hz <= 0:
            return 0
        row = round(self.params.bins_per_octave * np.log2(hz / self.params.fmin))
        return int(np.clip(row, 0, self.params.n_bins - 1))

    def row_to_value(self, row: int, sr: int) -> float:
        return float(self.params.fmin * 2 ** (row / self.params.bins_per_octave))

    # CQT rows are log-spaced, so the display axis is bin-index with note tick labels.
    def row_to_display_y(self, row: int, sr: int) -> float:
        return float(row)

    def y_extent(self, coeffs: np.ndarray, sr: int) -> tuple[float, float]:
        return (0.0, float(coeffs.shape[0]))

    def y_ticks(self, coeffs: np.ndarray, sr: int) -> tuple[list, list]:
        positions = list(range(0, self.params.n_bins, self.params.bins_per_octave))
        labels = [librosa.hz_to_note(self.row_to_value(p, sr)) for p in positions]
        return (positions, labels)

    def provenance(self) -> dict:
        # Both the CQT view params and the STFT reconstruction params are recorded:
        # the audio really is produced via this STFT (so fft_size/hop/window are true).
        return {
            "fft_size": self.stft_params.n_fft,
            "hop_length": self.stft_params.hop_length,
            "window_type": self.stft_params.window,
            "cqt_fmin": self.params.fmin,
            "cqt_n_bins": self.params.n_bins,
            "cqt_bins_per_octave": self.params.bins_per_octave,
        }
