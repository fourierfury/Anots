"""Entry point: ``python -m python_ref.gui <audio-file>``."""

from __future__ import annotations

import sys

from .app import run

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m python_ref.gui <audio-file>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(sys.argv[1]))
