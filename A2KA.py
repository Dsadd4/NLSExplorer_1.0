"""Compatibility module for existing ``from A2KA import ...`` notebooks.

Install the repository first with ``python -m pip install -e .``. New code may
import the same public objects directly from :mod:`nlsexplorer`.
"""

from nlsexplorer import A2KA, SCNLS

__all__ = ["A2KA", "SCNLS"]
