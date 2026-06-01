"""Canonical spectral transforms.

Thin librosa wrappers with the locked parameters applied. Each returns exactly what
the native core must reproduce within ``params.TOLERANCE``.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..params import DB, MEL, MFCC, STFT, MelParams, MfccParams, StftParams


def stft(y: np.ndarray, params: StftParams = STFT) -> np.ndarray:
    """Complex STFT. Magnitude and phase are preserved so the inverse is exact."""
    return librosa.stft(
        y,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window=params.window,
        center=params.center,
    )


def istft(stft_matrix: np.ndarray, params: StftParams = STFT, length: int | None = None) -> np.ndarray:
    """Inverse STFT. Pass ``length`` to recover the exact original sample count."""
    return librosa.istft(
        stft_matrix,
        hop_length=params.hop_length,
        window=params.window,
        center=params.center,
        length=length,
    )


def amplitude_to_db(magnitude: np.ndarray, params: DbParams = DB) -> np.ndarray:
    """Magnitude spectrogram to dB (ref = peak), matching Audacity's display."""
    return librosa.amplitude_to_db(magnitude, ref=np.max, top_db=params.top_db)


def melspectrogram(
    y: np.ndarray,
    sr: int,
    stft_params: StftParams = STFT,
    mel_params: MelParams = MEL,
) -> np.ndarray:
    """Power mel spectrogram, ``(n_mels, n_frames)``, Slaney filterbank."""
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
    """MFCC matrix, ``(n_mfcc, n_frames)``, from the log-power Slaney mel spectrogram."""
    log_mel = librosa.power_to_db(melspectrogram(y, sr, stft_params, mel_params))
    return librosa.feature.mfcc(
        S=log_mel,
        sr=sr,
        n_mfcc=mfcc_params.n_mfcc,
        dct_type=mfcc_params.dct_type,
        norm=mfcc_params.norm,
    )
