"""Transform domains: the Qt-free analysis/reconstruction/coordinate core.

Adding a new view (CQT, chroma, scalogram) means adding one module here and one
registry entry below — no changes to the session or GUI shell.
"""

from .base import Transform
from .chroma import ChromaTransform
from .cqt import CqtTransform
from .scalogram import ScalogramTransform
from .stft import StftTransform

#: name -> Transform subclass, for view selection in the GUI
TRANSFORMS: dict[str, type[Transform]] = {
    StftTransform.name: StftTransform,
    CqtTransform.name: CqtTransform,
    ChromaTransform.name: ChromaTransform,
    ScalogramTransform.name: ScalogramTransform,
}

__all__ = [
    "Transform", "StftTransform", "CqtTransform", "ChromaTransform",
    "ScalogramTransform", "TRANSFORMS",
]
