"""Entry point: ``python -m python_ref.gui <audio-file> [--view VIEW]``."""

from __future__ import annotations

import argparse

from ..transforms import TRANSFORMS
from .app import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m python_ref.gui")
    parser.add_argument("audio_file", help="path to a WAV/FLAC/etc. file to annotate")
    parser.add_argument(
        "--view", default="stft", choices=sorted(TRANSFORMS),
        help="transform-domain view to open (default: stft)",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.audio_file, args.view))
