"""Run the reproducible analytic-versus-Quadpy comparison benchmark.

    PYTHONPATH=src .venv/bin/python benchmarks/legacy/compare_stokes_3d.py --quick --plot

The example owns the shared implementation, CLI, isolated workers and outputs.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/legacy/compare_stokes_3d.py"
        ),
        run_name="__main__",
    )
