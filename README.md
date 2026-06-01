# Anots

**A domain-agnostic acoustic annotation workstation for building ML audio datasets.**

Anots turns raw recordings into clean, labeled, model-ready training data. You select
events directly in the time–frequency domain — on a spectrogram, mel spectrogram, MFCC,
CQT, and other transform views — and Anots reconstructs the selected audio, normalizes
it, and exports both the clip and its feature matrix with full provenance.

Every view Anots renders is **scientifically identical** to the same view in librosa or
Audacity given matching parameters. It is a precision scientific instrument, not a
creative visualizer.

## Why a Python reference first

This repository begins with a **Python reference prototype** (`python_ref/`) built on
librosa. It serves two purposes:

1. **A working annotation tool immediately** — usable for any acoustic domain
   (bioacoustics, medical audio, industrial monitoring, speech, music, …).
2. **Ground truth for the future native core.** A later Rust + C++ (FFTW3) engine will be
   validated against this reference: every DSP output must match the librosa reference
   within `1e-5`. The Python reference *defines* what "correct" means.

## Status

Early development. Current focus: the DSP reference core and its validation harness.

| Component | State |
|-----------|-------|
| DSP reference core (STFT, ISTFT, mel, MFCC) | Done |
| Validation harness (`1e-5` comparison, round-trip, analytic self-checks) | Done |
| Annotation model, masks, reconstruction, padding, sidecar export | Done |
| GUI (PyQt6 + matplotlib): spectrogram, rectangle select, playback, export | Done |
| Native Rust + C++ core | Planned |

## Locked DSP parameters

The reference parameters are fixed so feature output is reproducible and comparable
across the whole pipeline. See [`python_ref/params.py`](python_ref/params.py).

| Parameter | Value |
|-----------|-------|
| Window | Hann |
| FFT size | 1024 (default; 256–4096 configurable) |
| Hop length | 512 |
| Mel filterbank | Slaney (`htk=False`) |
| Mel bins | 128, range 0 Hz → sr/2 |
| MFCC | 40 coefficients, DCT type-2 (orthonormal) |
| Match tolerance | `1e-5` |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate the reference test signals
python tests/reference_signals/generate.py

# Run the validation suite
pytest

# Launch the annotation GUI (needs the optional GUI deps)
pip install -e ".[gui]"
python -m python_ref.gui path/to/audio.wav
```

## License

[MIT](LICENSE).
