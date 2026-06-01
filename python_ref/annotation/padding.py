"""Silence padding to a fixed length for batched model input (design §9).

Padding is always done in the time domain (zeros on the waveform), never by patching
the spectrogram, so the padded clip's spectrogram is identical to any other tool's.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Standard target durations (seconds) for common model families (design §9.2).
PRESETS = {"yamnet": 0.96, "vggish": 0.96, "2s": 2.0, "4s": 4.0, "panns": 10.0}


class PaddingMode(str, Enum):
    END = "end"
    START = "start"
    CENTER = "center"
    OFFSET = "offset"  # caller-supplied offset (e.g. random augmentation)


@dataclass(frozen=True)
class PaddingRecord:
    original_duration_samples: int
    padded_duration_samples: int
    padding_mode: str
    signal_offset_samples: int
    padding_target_source: str
    sample_rate: int

    @property
    def original_duration_ms(self) -> float:
        return self.original_duration_samples / self.sample_rate * 1000.0

    @property
    def padded_duration_ms(self) -> float:
        return self.padded_duration_samples / self.sample_rate * 1000.0


def preset_samples(name: str, sr: int) -> int:
    return round(PRESETS[name] * sr)


def _offset(mode: PaddingMode, pad_total: int, offset: int | None) -> int:
    if mode is PaddingMode.END:
        return 0
    if mode is PaddingMode.START:
        return pad_total
    if mode is PaddingMode.CENTER:
        return pad_total // 2
    if mode is PaddingMode.OFFSET:
        if offset is None or not 0 <= offset <= pad_total:
            raise ValueError(f"OFFSET mode needs offset in [0, {pad_total}], got {offset}")
        return offset
    raise ValueError(f"unknown padding mode: {mode}")


def pad(
    y: np.ndarray,
    target_samples: int,
    mode: PaddingMode = PaddingMode.END,
    sr: int = 44_100,
    target_source: str = "manual",
    offset: int | None = None,
) -> tuple[np.ndarray, PaddingRecord]:
    """Pad ``y`` with silence to ``target_samples``; returns the clip and its record."""
    if target_samples < len(y):
        raise ValueError(f"target {target_samples} shorter than signal {len(y)}")

    pad_total = target_samples - len(y)
    signal_offset = _offset(mode, pad_total, offset)

    out = np.zeros(target_samples, dtype=y.dtype)
    out[signal_offset : signal_offset + len(y)] = y

    record = PaddingRecord(
        original_duration_samples=len(y),
        padded_duration_samples=target_samples,
        padding_mode=mode.value,
        signal_offset_samples=signal_offset,
        padding_target_source=target_source,
        sample_rate=sr,
    )
    return out, record
