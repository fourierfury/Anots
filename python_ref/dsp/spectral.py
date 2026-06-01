"""Canonical spectral transforms.

Each function returns exactly what the corresponding librosa call returns with the
Anots locked parameters applied. The native core is validated against these.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..params import DB, MEL, MFCC, STFT, StftParams, MelParams, MfccParams


def stft(
    y: np.ndarray,
    params: StftParams = STFT,
) -> np.ndarray:
    """Complex STFT matrix, shape ``(1 + n_fft // 2, n_frames)``.

    Magnitude *and* phase are preserved so the inverse is exact.
    """
    return librosa.stft(
        y,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window=params.window,
        center=params.center,
    )


def istft(
    stft_matrix: np.ndarray,
    params: StftParams = STFT,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT. Overlap-add reconstruction of the time-domain signal.

    Pass ``length`` (original sample count) to recover the exact signal length;
    librosa otherwise trims to whole frames.
    """
    return librosa.istft(
        stft_matrix,
        hop_length=params.hop_length,
        window=params.window,
        center=params.center,
        length=length,
    )


def amplitude_to_db(magnitude: np.ndarray, params=DB) -> np.ndarray:
    """Convert a magnitude (|X|) spectrogram to dB: ``20 * log10`` with a floor.

    Uses ``librosa.amplitude_to_db`` (ref = peak) so output matches Audacity's dB
    display. The floor is ``-top_db`` relative to the peak.
    """
    return librosa.amplitude_to_db(magnitude, ref=np.max, top_db=params.top_db)


def melspectrogram(
    y: np.ndarray,
    sr: int,
    stft_params: StftParams = STFT,
    mel_params: MelParams = MEL,
) -> np.ndarray:
    """Mel spectrogram (power), shape ``(n_mels, n_frames)``.

    Uses the **Slaney** filterbank (``htk=False``). ``fmax=None`` means ``sr/2``.
    """
    return librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=stft_params.n_fft,
        hop_length=stft_params.hop_length,
        window=stft_params.window,
        center=stft_params.center,
        n_mels=mel_params.n_mels,
        htk=mel_params.htk,
        fmin=mel_params.fmin,
        fmax=mel_params.fmax,
    )


def mfcc(
    y: np.ndarray,
    sr: int,
    stft_params: StftParams = STFT,
    mel_params: MelParams = MEL,
    mfcc_params: MfccParams = MFCC,
) -> np.ndarray:
    """MFCC matrix, shape ``(n_mfcc, n_frames)``.

    DCT type-2, orthonormal, computed from the log-power Slaney mel spectrogram.
    Coefficient 0 is the DC / energy term.
    """
    mel_power = melspectrogram(y, sr, stft_params, mel_params)
    log_mel = librosa.power_to_db(mel_power)
    return librosa.feature.mfcc(
        S=log_mel,
        sr=sr,
        n_mfcc=mfcc_params.n_mfcc,
        dct_type=mfcc_params.dct_type,
        norm=mfcc_params.norm,
    )
