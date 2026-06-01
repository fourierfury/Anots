"""Locked DSP reference parameters.

This module is the single source of truth for the DSP parameters the whole Anots
pipeline must agree on. The Python reference (librosa) uses these values, and the
future native Rust + C++ core must match the Python output within ``TOLERANCE``.

These values are *locked*: do not change them without an explicit, documented
decision. Silent drift here (e.g. Slaney vs HTK mel, DCT type 2 vs 3) produces
feature matrices that look right but are subtly wrong, which makes training data
unusable.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Numerical agreement -----------------------------------------------------

#: Maximum allowed elementwise deviation between the Python reference and any
#: other implementation (e.g. the native core). Anything larger is a bug, never
#: a tolerance to be relaxed.
TOLERANCE: float = 1e-5

# --- Sample rates ------------------------------------------------------------

#: Reference / training sample rate. Preserves high-frequency content (to ~22 kHz
#: Nyquist) for downsampling experiments. Deployment inference may run lower.
TRAINING_SR: int = 44_100


@dataclass(frozen=True)
class StftParams:
    """Short-Time Fourier Transform parameters.

    Defaults match Audacity's default spectrogram and ``librosa.stft`` defaults
    (Hann window, centered framing). ``n_fft`` and ``hop_length`` are
    user-configurable per session but locked for a given dataset profile.
    """

    n_fft: int = 1024
    hop_length: int = 512
    window: str = "hann"
    center: bool = True

    def __post_init__(self) -> None:
        if self.n_fft not in (256, 512, 1024, 2048, 4096):
            raise ValueError(
                f"n_fft must be one of 256/512/1024/2048/4096, got {self.n_fft}"
            )
        if self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ValueError(
                f"hop_length must be in (0, n_fft]; got {self.hop_length} for "
                f"n_fft={self.n_fft}"
            )


@dataclass(frozen=True)
class MelParams:
    """Mel spectrogram parameters.

    ``htk=False`` selects the **Slaney** mel filterbank — librosa's default and the
    ML-community standard for non-speech audio. The HTK variant is NOT
    interchangeable and is reserved for explicit speech-recognition collaborations.
    """

    n_mels: int = 128
    htk: bool = False  # Slaney. Locked decision.
    fmin: float = 0.0
    #: fmax of None means sr/2 (full range, not 20 Hz–8 kHz).
    fmax: float | None = None


@dataclass(frozen=True)
class MfccParams:
    """MFCC parameters. Coefficient 0 is the DC / energy term."""

    n_mfcc: int = 40
    dct_type: int = 2  # librosa default; orthonormal via norm below.
    norm: str = "ortho"


@dataclass(frozen=True)
class DbParams:
    """dB magnitude scaling: ``20 * log10(|X|)`` with a floor, as Audacity displays."""

    top_db: float = 120.0  # floor at -120 dB relative to peak.


# --- Default singletons ------------------------------------------------------

STFT = StftParams()
MEL = MelParams()
MFCC = MfccParams()
DB = DbParams()
