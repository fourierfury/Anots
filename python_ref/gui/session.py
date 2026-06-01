"""Qt-free annotation session: loaded audio, its transform, layers, and operations.

The session is transform-agnostic: it holds a :class:`~python_ref.transforms.Transform`
(STFT by default) and routes every domain-specific operation — forward analysis,
display, reconstruction, coordinate mapping, provenance — through it. Swapping in a
CQT or scalogram transform changes none of the code below.
"""

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
    OverlapWarning,
    Tool,
    detect_overlaps,
    energy_brush,
    feather,
    layer_color,
    pad,
    polygon_mask,
    rectangle_mask,
    ridge_path,
    tube_mask,
    write_csv,
    write_sidecar,
)
from ..annotation.brush import DEFAULT_RADIUS as BRUSH_RADIUS
from ..annotation.ridge import MAX_JUMP, SMOOTHNESS, TUBE_WIDTH
from ..annotation.padding import PaddingMode
from ..params import STFT, StftParams
from ..transforms import StftTransform, Transform


class SpectrogramSession:
    def __init__(
        self,
        y: np.ndarray,
        sr: int,
        source_file: str = "",
        params: StftParams = STFT,
        transform: Transform | None = None,
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.sr = sr
        self.source_file = source_file
        self.transform = transform or StftTransform(params)
        self.params = params  # back-compat: STFT params for the dataset profile
        self.coeffs = self.transform.forward(self.y, sr)
        self.layers: list[Layer] = []

    @property
    def stft(self) -> np.ndarray:
        """Back-compat alias for the coefficient matrix (STFT when that transform)."""
        return self.coeffs

    @classmethod
    def load(
        cls,
        path: str | Path,
        params: StftParams = STFT,
        transform: Transform | None = None,
    ) -> "SpectrogramSession":
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return cls(y, int(sr), source_file=str(path), params=params, transform=transform)

    @property
    def duration_s(self) -> float:
        return len(self.y) / self.sr

    def magnitude_db(self) -> np.ndarray:
        """dB-scaled magnitude spectrogram for display."""
        return self.transform.display_db(self.coeffs)

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
        """Create a feathered rectangular layer from a time / **Hz** selection."""
        bins = (self.transform.value_to_row(f0_hz, self.sr),
                self.transform.value_to_row(f1_hz, self.sr))
        return self._add_rectangle(t0, t1, bins, label, label_class, confidence,
                                   extraction_mode, taper_bins)

    def add_rectangle_display(
        self,
        t0: float,
        t1: float,
        y0: float,
        y1: float,
        label: str,
        label_class: str = "",
        confidence: float = 1.0,
        extraction_mode: ExtractionMode = ExtractionMode.POSITIVE,
        taper_bins: int = 8,
    ) -> Layer:
        """Create a layer from a selection in the view's **display** y-units.

        The GUI works in display coordinates: Hz on the STFT view, but a bin/band index
        on the log-scale views (CQT, chroma, scalogram). ``display_y_to_row`` resolves
        either to the right coefficient row, so one rectangle drag works in every view.
        """
        bins = (self.transform.display_y_to_row(y0, self.sr),
                self.transform.display_y_to_row(y1, self.sr))
        return self._add_rectangle(t0, t1, bins, label, label_class, confidence,
                                   extraction_mode, taper_bins)

    def _add_rectangle(
        self,
        t0: float,
        t1: float,
        bins: tuple[int, int],
        label: str,
        label_class: str,
        confidence: float,
        extraction_mode: ExtractionMode,
        taper_bins: int,
    ) -> Layer:
        frames = sorted((self.transform.time_to_col(t0, self.sr),
                         self.transform.time_to_col(t1, self.sr)))
        mask = rectangle_mask(self.coeffs.shape[0], self.coeffs.shape[1],
                              freq_bins=tuple(sorted(bins)), frame_bins=tuple(frames))
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

    def add_lasso(
        self,
        points: list[tuple[float, float]],
        label: str,
        label_class: str = "",
        confidence: float = 1.0,
        extraction_mode: ExtractionMode = ExtractionMode.POSITIVE,
        taper_bins: int = 8,
    ) -> Layer:
        """Freehand lasso: fill the closed outline drawn by ``points`` (design §6.4).

        ``points`` is the drag path in display units ``(seconds, display_y)``; it is mapped
        through the active transform to ``(frame, freq_bin)`` vertices and filled, so any
        shape can be drawn in any view.
        """
        verts = [(self.transform.time_to_col(t, self.sr), self.transform.display_y_to_row(y, self.sr))
                 for t, y in points]
        mask = polygon_mask(self.coeffs.shape[0], self.coeffs.shape[1], verts)
        layer = Layer(
            mask=feather(mask, taper_bins),
            label=label,
            label_class=label_class,
            confidence=confidence,
            tool_used=Tool.LASSO,
            extraction_mode=extraction_mode,
            color=layer_color(len(self.layers)),
        )
        self.layers.append(layer)
        return layer

    def add_ridge(
        self,
        t0: float,
        f0_hz: float,
        t1: float,
        f1_hz: float,
        label: str,
        label_class: str = "",
        confidence: float = 1.0,
        extraction_mode: ExtractionMode = ExtractionMode.POSITIVE,
        k: int = MAX_JUMP,
        lam: float = SMOOTHNESS,
        width: int = TUBE_WIDTH,
    ) -> Layer:
        """Follow the maximum-energy ridge between two points given in **Hz** (design §6.5)."""
        start = (self.transform.time_to_col(t0, self.sr), self.transform.value_to_row(f0_hz, self.sr))
        end = (self.transform.time_to_col(t1, self.sr), self.transform.value_to_row(f1_hz, self.sr))
        return self._add_ridge(start, end, label, label_class, confidence,
                               extraction_mode, k, lam, width)

    def add_ridge_display(
        self,
        t0: float,
        y0: float,
        t1: float,
        y1: float,
        label: str,
        label_class: str = "",
        confidence: float = 1.0,
        extraction_mode: ExtractionMode = ExtractionMode.POSITIVE,
        k: int = MAX_JUMP,
        lam: float = SMOOTHNESS,
        width: int = TUBE_WIDTH,
    ) -> Layer:
        """Ridge between two points in the view's **display** y-units (Hz on STFT, band
        index on the log views) — what a GUI click provides. See :meth:`add_ridge`."""
        start = (self.transform.time_to_col(t0, self.sr), self.transform.display_y_to_row(y0, self.sr))
        end = (self.transform.time_to_col(t1, self.sr), self.transform.display_y_to_row(y1, self.sr))
        return self._add_ridge(start, end, label, label_class, confidence,
                               extraction_mode, k, lam, width)

    def _add_ridge(self, start, end, label, label_class, confidence,
                   extraction_mode, k, lam, width) -> Layer:
        path = ridge_path(np.abs(self.coeffs), start, end, k=k, lam=lam)
        mask = tube_mask(self.coeffs.shape, path, width=width)
        layer = Layer(
            mask=mask,
            label=label,
            label_class=label_class,
            confidence=confidence,
            tool_used=Tool.RIDGE,
            extraction_mode=extraction_mode,
            color=layer_color(len(self.layers)),
        )
        self.layers.append(layer)
        return layer

    def add_brush(
        self,
        points: list[tuple[float, float]],
        label: str,
        radius: int = BRUSH_RADIUS,
        threshold_db: float | None = None,
        label_class: str = "",
        confidence: float = 1.0,
        extraction_mode: ExtractionMode = ExtractionMode.POSITIVE,
        taper_bins: int = 8,
    ) -> Layer:
        """Paint a Smart Energy Brush stroke into a new layer (design §6.6).

        ``points`` is the drag path in display units ``(seconds, display_y)`` — Hz on the
        STFT view, band index on the log views. Each point stamps a circular brush of
        ``radius`` bins; ``threshold_db`` (if given) restricts the stamp to bins at or
        above that dB, so the brush grabs only the loud content.
        """
        db = self.transform.display_db(self.coeffs)
        mask = np.zeros(self.coeffs.shape, dtype=np.float64)
        for t, y in points:
            center = (self.transform.display_y_to_row(y, self.sr),
                      self.transform.time_to_col(t, self.sr))
            mask = energy_brush(db, center, radius, threshold_db, mask)
        layer = Layer(
            mask=feather(mask, taper_bins),
            label=label,
            label_class=label_class,
            confidence=confidence,
            tool_used=Tool.BRUSH,
            extraction_mode=extraction_mode,
            color=layer_color(len(self.layers)),
        )
        self.layers.append(layer)
        return layer

    def reconstruct_layer(self, layer: Layer) -> np.ndarray:
        return self.transform.reconstruct(
            self.y, self.coeffs, layer.mask, layer.extraction_mode, self.sr, length=len(self.y)
        )

    def overlaps(self, threshold: float = 0.0) -> list[OverlapWarning]:
        """Pairwise IoU overlap warnings across the current layers (design §5.4)."""
        return detect_overlaps(self.layers, threshold)

    def _bounds_samples(self, mask: np.ndarray) -> tuple[int, int]:
        """Sample range covered by a layer's active frames."""
        active = np.where(mask.any(axis=0))[0]
        if not len(active):
            return 0, len(self.y)
        start = self.transform.col_to_time(int(active[0]), self.sr)
        end = self.transform.col_to_time(int(active[-1]) + 1, self.sr)
        return int(start * self.sr), min(int(end * self.sr), len(self.y))

    def to_annotations(self) -> list[Annotation]:
        recon = self.transform.reconstruction()       # tagged {"method": ...} record
        view_params = self.transform.provenance()      # view-only display/selection params
        anns = []
        for layer in self.layers:
            start, end = self._bounds_samples(layer.mask)
            active_bins = np.where(layer.mask.any(axis=1))[0]
            f_lo = self.transform.row_to_value(int(active_bins[0]), self.sr) if len(active_bins) else 0.0
            f_hi = self.transform.row_to_value(int(active_bins[-1]), self.sr) if len(active_bins) else 0.0
            layer_meta = {**view_params, **self.transform.layer_params(layer.mask, self.sr)}
            anns.append(Annotation(
                source_file=self.source_file,
                sample_rate=self.sr,
                start_sample=start,
                end_sample=end,
                label=layer.label,
                label_class=layer.label_class,
                confidence=layer.confidence,
                view=self.transform.name,
                freq_min_hz=f_lo,
                freq_max_hz=f_hi,
                extraction_mode=layer.extraction_mode.value,
                reconstruction=recon,
                transform_params=layer_meta,
                notes=layer.notes,
                created_at=layer.created_at,
            ))
        return anns

    def export(
        self,
        out_dir: str | Path,
        pad_to: int | None = None,
        pad_mode: PaddingMode = PaddingMode.END,
        context_ms: float = 0.0,
    ) -> list[Annotation]:
        """Write each layer's clip (raw and optionally padded) plus a sidecar + CSV.

        The raw clip is **cropped to the event's time bounds** (the annotation's
        ``[start_sample, end_sample]``) — clean clip boundaries for ML extraction
        (design §10.5), not the whole recording. ``context_ms`` keeps an optional
        symmetric margin of surrounding audio (0 = sample-tight event). Padding, when
        requested, normalises the *cropped* clip's duration.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(self.source_file).stem or "clip"
        margin = int(round(context_ms / 1000.0 * self.sr))

        annotations = self.to_annotations()
        for i, (layer, ann) in enumerate(zip(self.layers, annotations)):
            full = self.reconstruct_layer(layer)
            lo = max(0, ann.start_sample - margin)
            hi = min(len(full), ann.end_sample + margin)
            clip = full[lo:hi]
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
