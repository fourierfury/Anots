"""Chromagram view (design §4.3.5).

Twelve pitch classes (C..B); energy at any octave folds into the same row. The view
is a tonal-background selector: select dominant pitch classes and subtract them to
suppress tonal contamination (hum, harmonic background) across the whole file.

**Reconstruction:** a selected pitch class spans every octave, so reconstruction is a
per-frame comb filter — keep exactly the STFT bins whose pitch class is selected, then
exact ISTFT. The comb realisation is exact (validated to ~1e-15 by linearity), but the
chroma *forward* is lossy w.r.t. octave, so the path as a whole is classed
``approximate`` per §11. Because a selection has no single frequency band (its audio is
octave-distributed, measured to span 220-880 Hz for a one-pitch-class selection), the
annotation records the pitch-class set rather than a freq range (design §4.3.5).
"""

from __future__ import annotations

import librosa
import numpy as np

from ..annotation.model import ExtractionMode
from ..annotation.reconstruct import effective_mask
from ..dsp import amplitude_to_db, istft, stft
from ..params import CHROMA, STFT, ChromaParams, StftParams
from .base import Transform

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class ChromaTransform(Transform):
    name = "chroma"
    accuracy_class = "approximate"  # comb realisation is exact; chroma forward is octave-lossy
    y_label = "Pitch class"

    def __init__(self, params: ChromaParams = CHROMA, stft_params: StftParams = STFT):
        self.params = params
        self.stft_params = stft_params

    def forward(self, y: np.ndarray, sr: int) -> np.ndarray:
        return librosa.feature.chroma_cqt(
            y=y, sr=sr,
            hop_length=self.params.hop_length,
            fmin=self.params.fmin,
            n_chroma=self.params.n_chroma,
            bins_per_octave=self.params.bins_per_octave,
        )

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
        S = stft(y, self.stft_params)
        stft_mask = self._mask_to_stft(mask, S.shape, sr)
        return istft(S * effective_mask(stft_mask, mode), self.stft_params, length=length)

    def _mask_to_stft(self, chroma_mask: np.ndarray, stft_shape: tuple[int, int], sr: int) -> np.ndarray:
        """Comb filter: each STFT bin takes the mask value of its pitch-class row."""
        n_bins_stft, n_frames_stft = stft_shape
        freqs = np.fft.rfftfreq(self.stft_params.n_fft, 1.0 / sr)
        valid = freqs > 0
        pc = np.zeros(n_bins_stft, dtype=int)
        pc[valid] = np.mod(np.round(librosa.hz_to_midi(freqs[valid])), 12).astype(int)

        n = min(n_frames_stft, chroma_mask.shape[1])
        stft_mask = np.zeros((n_bins_stft, n_frames_stft), dtype=np.float64)
        stft_mask[valid, :n] = chroma_mask[pc[valid][:, None], np.arange(n)]
        return stft_mask

    def time_to_col(self, t: float, sr: int) -> int:
        return round(t * sr / self.params.hop_length)

    def col_to_time(self, col: int, sr: int) -> float:
        return col * self.params.hop_length / sr

    def value_to_row(self, hz: float, sr: int) -> int:
        if hz <= 0:
            return 0
        return int(np.mod(round(librosa.hz_to_midi(hz)), 12))

    def row_to_value(self, row: int, sr: int) -> float:
        # A pitch class spans every octave — it has no single Hz. The annotation
        # records the pitch-class set instead (see layer_params); 0 is the N/A sentinel.
        return 0.0

    def row_to_display_y(self, row: int, sr: int) -> float:
        return float(row)

    def y_extent(self, coeffs: np.ndarray, sr: int) -> tuple[float, float]:
        return (0.0, float(self.params.n_chroma))

    def y_ticks(self, coeffs: np.ndarray, sr: int) -> tuple[list, list]:
        positions = [r + 0.5 for r in range(self.params.n_chroma)]
        return (positions, list(PITCH_CLASSES[: self.params.n_chroma]))

    def layer_params(self, mask: np.ndarray, sr: int) -> dict:
        active = np.where(mask.any(axis=1))[0]
        classes = sorted({PITCH_CLASSES[int(r) % 12] for r in active},
                         key=lambda c: PITCH_CLASSES.index(c))
        return {"pitch_classes": classes}

    def provenance(self) -> dict:
        return {
            "fft_size": self.stft_params.n_fft,
            "hop_length": self.stft_params.hop_length,
            "window_type": self.stft_params.window,
            "chroma_n_chroma": self.params.n_chroma,
            "chroma_bins_per_octave": self.params.bins_per_octave,
        }
