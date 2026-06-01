"""Locked DSP reference parameters.

Single source of truth for the parameters the whole pipeline agrees on. The native
core must reproduce the Python reference output within ``TOLERANCE``. Don't change
these without an explicit decision — silent drift (Slaney vs HTK mel, DCT type)
yields feature matrices that look right but are subtly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

TOLERANCE: float = 1e-5
TRAINING_SR: int = 44_100


@dataclass(frozen=True)
class StftParams:
    n_fft: int = 1024
    hop_length: int = 512
    window: str = "hann"
    center: bool = True

    def __post_init__(self) -> None:
        if self.n_fft not in (256, 512, 1024, 2048, 4096):
            raise ValueError(f"n_fft must be one of 256/512/1024/2048/4096, got {self.n_fft}")
        if not 0 < self.hop_length <= self.n_fft:
            raise ValueError(f"hop_length must be in (0, {self.n_fft}], got {self.hop_length}")


@dataclass(frozen=True)
class MelParams:
    n_mels: int = 128
    htk: bool = False  # Slaney filterbank — not interchangeable with HTK.
    fmin: float = 0.0
    fmax: float | None = None  # None => sr/2.


@dataclass(frozen=True)
class MfccParams:
    n_mfcc: int = 40
    dct_type: int = 2
    norm: str = "ortho"


@dataclass(frozen=True)
class DbParams:
    top_db: float = 120.0  # floor at -120 dB below peak.


STFT = StftParams()
MEL = MelParams()
MFCC = MfccParams()
DB = DbParams()
