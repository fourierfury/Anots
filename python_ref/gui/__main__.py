"""Entry point: ``python -m python_ref.gui [audio-file] [--view VIEW]``.

The audio file is optional — launch with no argument and load a file from your system via
the **Open audio…** button.
"""

from __future__ import annotations

import argparse

from ..transforms import TRANSFORMS
from .app import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m python_ref.gui")
    parser.add_argument("audio_file", nargs="?", default=None,
                        help="optional path to a WAV/FLAC/etc. file (else use 'Open audio…')")
    parser.add_argument(
        "--view", default="stft", choices=sorted(TRANSFORMS),
        help="transform-domain view to open (default: stft)",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.audio_file, args.view))
