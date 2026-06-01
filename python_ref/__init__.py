"""Anots Python reference prototype.

A librosa-based reference implementation of the Anots DSP pipeline. It is both a
usable annotation back end and the numerical ground truth against which the future
native (Rust + C++) core is validated. See :mod:`python_ref.params` for the locked
parameters and :mod:`python_ref.validation` for the comparison harness.
"""

from . import dsp, params, validation

__all__ = ["dsp", "params", "validation"]
__version__ = "0.0.1"
