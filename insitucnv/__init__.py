"""Top-level package for insitucnv.

The package deliberately avoids importing heavy optional dependencies at import
time so that lightweight utilities, such as the quantitative validation
framework, remain usable in notebook environments with only a subset of the
full stack installed.
"""

__all__ = ["pp", "tl"]
