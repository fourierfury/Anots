"""Annotation data model: layers and export records (design §8.2, §10.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

import numpy as np

from ..params import STFT

# ColorBrewer Set1 — chosen for distinguishable overlapping regions (design §5.2).
LAYER_COLORS = (
    "#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00",
    "#984EA3", "#A65628", "#F781BF", "#999999",
)


class Tool(str, Enum):
    RECTANGLE = "rectangle"
    SPLINE = "spline"
    LASSO = "lasso"
    RIDGE = "ridge"
    HARMONIC_COMB = "harmonic_comb"
    BRUSH = "brush"


class ExtractionMode(str, Enum):
    POSITIVE = "positive"  # keep the selection
    NEGATIVE = "negative"  # keep everything except the selection (hard negative)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def layer_color(index: int) -> str:
    return LAYER_COLORS[index % len(LAYER_COLORS)]


@dataclass
class Layer:
    """A single annotation: a time-frequency mask plus its label and provenance."""

    mask: np.ndarray  # (n_freq, n_frames), float in [0, 1]
    label: str
    label_class: str = ""
    confidence: float = 1.0
    tool_used: Tool = Tool.RECTANGLE
    extraction_mode: ExtractionMode = ExtractionMode.POSITIVE
    color: str = LAYER_COLORS[0]
    visible: bool = True
    notes: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utcnow_iso)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class Annotation:
    """Sample-accurate export record (design §10.1).

    Timestamps live as integer sample indices; all time units derive from them, so
    no floating-point error accumulates across conversions (design §10.2).
    """

    source_file: str
    sample_rate: int
    start_sample: int
    end_sample: int
    label: str
    label_class: str = ""
    confidence: float = 1.0
    view: str = "stft"
    freq_min_hz: float = 0.0
    freq_max_hz: float = 0.0
    mask_type: str = "rectangle"
    extraction_mode: str = ExtractionMode.POSITIVE.value
    fft_size: int = STFT.n_fft
    hop_length: int = STFT.hop_length
    window_type: str = STFT.window
    n_mels: int | None = None
    annotator: str = ""
    notes: str = ""
    exported_clip_raw: str | None = None
    exported_clip_padded: str | None = None
    feature_export: str | None = None
    created_at: str = field(default_factory=utcnow_iso)

    @property
    def start_s(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_s(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def start_ms(self) -> float:
        return self.start_s * 1000.0

    @property
    def end_ms(self) -> float:
        return self.end_s * 1000.0

    @property
    def start_us(self) -> float:
        return self.start_s * 1_000_000.0

    def to_dict(self) -> dict:
        """Flat record with derived time units, suitable for JSON/CSV export."""
        return {
            "source_file": self.source_file,
            "sample_rate": self.sample_rate,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "start_us": round(self.start_us, 1),
            "label": self.label,
            "label_class": self.label_class,
            "confidence": self.confidence,
            "view": self.view,
            "freq_min_hz": self.freq_min_hz,
            "freq_max_hz": self.freq_max_hz,
            "mask_type": self.mask_type,
            "extraction_mode": self.extraction_mode,
            "fft_size": self.fft_size,
            "hop_length": self.hop_length,
            "window_type": self.window_type,
            "n_mels": self.n_mels,
            "annotator": self.annotator,
            "notes": self.notes,
            "exported_clip_raw": self.exported_clip_raw,
            "exported_clip_padded": self.exported_clip_padded,
            "feature_export": self.feature_export,
            "created_at": self.created_at,
        }
