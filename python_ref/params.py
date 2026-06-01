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
class CqtParams:
    """Constant-Q transform (design §4.3.4). One bin per semitone.

    ``hop_length`` matches ``StftParams`` so CQT and STFT frames align: a selection
    drawn on the CQT maps frame-for-frame onto the STFT for exact reconstruction
    (the reference's CQT reconstruction path; see ``transforms.cqt``).
    """

    fmin: float = 32.70319566257483  # C1
    n_bins: int = 84  # 7 octaves
    bins_per_octave: int = 12  # one bin = one semitone
    hop_length: int = 512


@dataclass(frozen=True)
class ChromaParams:
    """Chromagram (design §4.3.5). Energy folded into 12 pitch classes.

    ``hop_length`` matches ``StftParams`` so a pitch-class selection maps frame-for-frame
    onto the STFT comb filter used for reconstruction (see ``transforms.chroma``).
    """

    fmin: float = 32.70319566257483  # C1
    n_chroma: int = 12
    bins_per_octave: int = 36
    hop_length: int = 512


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
CQT = CqtParams()
CHROMA = ChromaParams()
MFCC = MfccParams()
DB = DbParams()
