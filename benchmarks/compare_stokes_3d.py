"""Run the reproducible analytic-versus-Quadpy comparison benchmark.

    PYTHONPATH=src .venv/bin/python benchmarks/compare_stokes_3d.py --quick --plot

The example owns the shared implementation, CLI, isolated workers and outputs.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "compare_stokes_3d.py"),
        run_name="__main__",
    )
