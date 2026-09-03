"""Resolve the training package root and add it to ``sys.path``.

Every script under ``training/scripts/`` (at any nesting depth) imports this
module to locate the training package root -- the directory containing
``config.py``.  This avoids fragile chains of ``.parent.parent.parent`` that
break whenever the directory structure changes.

Usage (at the top of any script, before importing project modules)::

    import _resolve  # noqa: F401  (adds training root to sys.path)

After this import, ``import config``, ``from models import ...``, etc. work
regardless of where the script sits in the ``scripts/`` tree.
"""

import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file until we find the directory containing ``config.py``."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # safety limit
        if (current / "config.py").is_file():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not find training package root (directory containing config.py). "
        "Searched upward from: " + str(Path(__file__).resolve().parent)
    )


if str(_find_project_root()) not in sys.path:
    sys.path.insert(0, str(_find_project_root()))
