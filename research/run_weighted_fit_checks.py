"""Execute just the numerical fit-order comparison and save notebook outputs."""

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "research/weighted_stokeslet_analytic.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    original_cwd = Path.cwd()
    import os

    os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # Setup, kernel definitions and independent quadrature helpers only.
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if (
                    (
                        "import platform" in source
                        and "SYMBOLIC_TIMEOUT_SECONDS =" in source
                    )
                    or source.startswith("r = sp.Symbol")
                    or "def polar_velocity(" in source
                ):
                    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - Execute selected local notebook cells.
                    namespace["display"] = lambda *args: None
        cell = next(
            c for c in notebook["cells"] if c.get("id") == "fit-order-comparison-code"
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)  # noqa: S102 - Execute the local comparison cell.
        output = buffer.getvalue()
        cell["execution_count"] = (
            None  # External runner, not a kernel execution counter.
        )
        cell["outputs"] = [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": output.splitlines(keepends=True),
            }
        ]
        cell["metadata"]["executed_by"] = "research/run_weighted_fit_checks.py"
        notebook["metadata"]["fit_comparison_python"] = sys.version
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(output, end="")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
