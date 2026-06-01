"""Parameter-locked DSP reference functions (thin librosa wrappers)."""

from .spectral import amplitude_to_db, istft, melspectrogram, mfcc, stft

__all__ = ["stft", "istft", "amplitude_to_db", "melspectrogram", "mfcc"]
