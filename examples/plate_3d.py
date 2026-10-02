"""Compact KL 3D/2D spectrum and Q comparison; see formulation_comparison.py.

Run: PYTHONPATH=src python examples/plate_3d.py --resonances 1
For an odd mode supply --symmetry antisymmetric --f-min ... --f-max ...
"""

import sys

from formulation_comparison import main

if __name__ == "__main__":
    main(["--models", "KL_3D", "KL_2D", "--output", "results/plate_3d", *sys.argv[1:]])
