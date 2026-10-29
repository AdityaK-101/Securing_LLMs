"""Split protocols used by the evaluation.

In-distribution (mixed)
    ``data/splits/{train,val,test}.csv`` from ``data/cleaned/combined_dataset.csv``.
    v3.0 grows the old 350-prompt corpus from raw sources (~700 rows;
    ~100-row test). Rebuild: ``python scripts/build_combined_dataset.py``.
    v2.0 350 archived under ``data/cleaned/legacy/`` and
    ``data/splits/legacy_v2.0_350/``.

Source holdout (mentor)
    ``data/splits/source_holdout/<source>/{train,val,test}.csv``
    Built by ``notebooks/build_source_holdout.ipynb`` (CLI wrapper:
    ``scripts/build_source_holdout.py``). Fully uniform four-source folds
    (520/130/400); Neuralchemy is the fifth fold.
"""

from .combined import build_combined_corpus, build_combined_dataset
from .source_holdout import (
    ATTACK_SOURCES,
    EXTERNAL_SOURCE,
    build_source_holdout,
    load_source_holdout,
)

__all__ = [
    "ATTACK_SOURCES",
    "EXTERNAL_SOURCE",
    "build_combined_corpus",
    "build_combined_dataset",
    "build_source_holdout",
    "load_source_holdout",
]
