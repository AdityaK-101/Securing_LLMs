"""CLI wrapper: execute notebooks/build_source_holdout.ipynb code cells.

The notebook is the canonical holdout builder. This script only exists so
automation can run without opening Jupyter:

  python scripts/build_source_holdout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "build_source_holdout.ipynb"


def main() -> None:
    if not NB.is_file():
        raise SystemExit(f"Missing notebook: {NB}")

    # Headless matplotlib when not inside Jupyter.
    import matplotlib

    matplotlib.use("Agg")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    # Match notebook assumption: cwd / ROOT on path; notebook uses Path("..").
    # Execute with ROOT as the project root by chdir to notebooks/.
    import os

    notebooks_dir = NB.parent
    prev = Path.cwd()
    os.chdir(notebooks_dir)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        ns: dict = {"__name__": "__main__"}
        # Provide display() when IPython is absent.
        try:
            from IPython.display import display  # noqa: F401
        except ImportError:
            ns["display"] = print

        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            exec(compile(src, str(NB), "exec"), ns, ns)
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    main()
