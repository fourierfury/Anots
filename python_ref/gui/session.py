"""Qt-free annotation session: loaded audio, its STFT, layers, and operations."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..annotation import (
    Annotation,
    DatasetProfile,
    ExtractionMode,
    Layer,
    Tool,
    feather,
    layer_color,
    pad,
    reconstruct,
    rectangle_mask,
    write_csv,
    write_sidecar,
)
from ..annotation.padding import PaddingMode
from ..dsp import amplitude_to_db, stft
from ..params import STFT, StftParams
from . import coords


class SpectrogramSession:
    def __init__(self, y: np.ndarray, sr: int, source_file: str = "", params: StftParams = STFT):
        self.y = np.asarray(y, dtype=np.float64)
        self.sr = sr
        self.source_file = source_file
        self.params = params
        self.stft = stft(self.y, params)
        self.layers: list[Layer] = []

    @classmethod
    def load(cls, path: str | Path, params: StftParams = STFT) -> "SpectrogramSession":
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return cls(y, int(sr), source_file=str(path), params=params)

    @property
    def duration_s(self) -> float:
        return len(self.y) / self.sr

    def magnitude_db(self) -> np.ndarray:
        """dB-scaled magnitude spectrogram for display."""
        return amplitude_to_db(np.abs(self.stft))

    def add_rectangle(
        self,
        t0: float,
        t1: float,
        f0_hz: float,
        f1_hz: float,
        label: str,
        label_class: str = "",
        confidence: float = 1.0,
        extraction_mode: ExtractionMode = ExtractionMode.POSITIVE,
        taper_bins: int = 8,
    ) -> Layer:
        """Create a feathered rectangular layer from a time/frequency selection."""
        frames = sorted((coords.time_to_frame(t0, self.sr, self.params),
                         coords.time_to_frame(t1, self.sr, self.params)))
        bins = sorted((coords.hz_to_bin(f0_hz, self.sr, self.params),
                       coords.hz_to_bin(f1_hz, self.sr, self.params)))

        mask = rectangle_mask(self.stft.shape[0], self.stft.shape[1],
                              freq_bins=tuple(bins), frame_bins=tuple(frames))
        layer = Layer(
            mask=feather(mask, taper_bins),
            label=label,
            label_class=label_class,
            confidence=confidence,
            tool_used=Tool.RECTANGLE,
            extraction_mode=extraction_mode,
            color=layer_color(len(self.layers)),
        )
        self.layers.append(layer)
        return layer

    def reconstruct_layer(self, layer: Layer) -> np.ndarray:
        return reconstruct(self.stft, layer.mask, layer.extraction_mode, self.params, length=len(self.y))

    def _bounds_samples(self, mask: np.ndarray) -> tuple[int, int]:
        """Sample range covered by a layer's active frames."""
        active = np.where(mask.any(axis=0))[0]
        if not len(active):
            return 0, len(self.y)
        start = coords.frame_to_time(int(active[0]), self.sr, self.params)
        end = coords.frame_to_time(int(active[-1]) + 1, self.sr, self.params)
        return int(start * self.sr), min(int(end * self.sr), len(self.y))

    def to_annotations(self) -> list[Annotation]:
        anns = []
        for layer in self.layers:
            start, end = self._bounds_samples(layer.mask)
            active_bins = np.where(layer.mask.any(axis=1))[0]
            f_lo = coords.bin_to_hz(int(active_bins[0]), self.sr, self.params) if len(active_bins) else 0.0
            f_hi = coords.bin_to_hz(int(active_bins[-1]), self.sr, self.params) if len(active_bins) else 0.0
            anns.append(Annotation(
                source_file=self.source_file,
                sample_rate=self.sr,
                start_sample=start,
                end_sample=end,
                label=layer.label,
                label_class=layer.label_class,
                confidence=layer.confidence,
                freq_min_hz=f_lo,
                freq_max_hz=f_hi,
                extraction_mode=layer.extraction_mode.value,
                fft_size=self.params.n_fft,
                hop_length=self.params.hop_length,
                window_type=self.params.window,
                notes=layer.notes,
                created_at=layer.created_at,
            ))
        return anns

    def export(
        self,
        out_dir: str | Path,
        pad_to: int | None = None,
        pad_mode: PaddingMode = PaddingMode.END,
    ) -> list[Annotation]:
        """Write each layer's clip (raw and optionally padded) plus a sidecar + CSV."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(self.source_file).stem or "clip"

        annotations = self.to_annotations()
        for i, (layer, ann) in enumerate(zip(self.layers, annotations)):
            clip = self.reconstruct_layer(layer)
            raw_path = out / f"{stem}_{i:03d}_raw.wav"
            sf.write(raw_path, clip.astype(np.float32), self.sr, subtype="FLOAT")
            ann.exported_clip_raw = str(raw_path)
            if pad_to is not None:
                padded, _ = pad(clip, pad_to, pad_mode, sr=self.sr)
                padded_path = out / f"{stem}_{i:03d}_padded.wav"
                sf.write(padded_path, padded.astype(np.float32), self.sr, subtype="FLOAT")
                ann.exported_clip_padded = str(padded_path)

        profile = DatasetProfile(sample_rate=self.sr, fft_size=self.params.n_fft,
                                 hop_length=self.params.hop_length, window_type=self.params.window)
        write_sidecar(out / stem, annotations, profile)
        write_csv(out / f"{stem}_annotations.csv", annotations)
        return annotations
