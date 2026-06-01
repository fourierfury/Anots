"""Annotation persistence: JSON sidecar, CSV, and dataset profile (design §10.3–10.7)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path

from ..params import STFT
from .model import Annotation

SIDECAR_SUFFIX = ".dspws.json"


@dataclass(frozen=True)
class DatasetProfile:
    """Locked FFT parameters for a session; mixing profiles yields incomparable features."""

    sample_rate: int
    fft_size: int = STFT.n_fft
    hop_length: int = STFT.hop_length
    window_type: str = STFT.window
    n_mels: int | None = None


def sidecar_path(audio_path: str | Path) -> Path:
    p = Path(audio_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def write_sidecar(
    audio_path: str | Path,
    annotations: list[Annotation],
    profile: DatasetProfile | None = None,
) -> Path:
    """Write annotations next to their audio file as ``<audio>.dspws.json``."""
    path = sidecar_path(audio_path)
    payload = {
        "profile": profile.__dict__ if profile else None,
        "annotations": [a.to_dict() for a in annotations],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_sidecar(audio_path: str | Path) -> tuple[DatasetProfile | None, list[Annotation]]:
    """Restore annotations from a sidecar; returns the profile and annotation list."""
    payload = json.loads(sidecar_path(audio_path).read_text())
    profile = DatasetProfile(**payload["profile"]) if payload.get("profile") else None
    return profile, [_annotation_from_dict(d) for d in payload["annotations"]]


def write_csv(path: str | Path, annotations: list[Annotation]) -> Path:
    """Flat one-row-per-annotation CSV for pandas/numpy pipelines (design §10.3)."""
    path = Path(path)
    rows = [a.to_dict() for a in annotations]
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _annotation_from_dict(d: dict) -> Annotation:
    # Derived time fields are recomputed from sample indices, so keep only init args.
    init_names = {f.name for f in fields(Annotation)}
    return Annotation(**{k: v for k, v in d.items() if k in init_names})
