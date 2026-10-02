"""Compatibility imports and benchmark entry point for the legacy hybrid model."""

from mufsi.hydrodynamics.legacy.stokeslet_hybrid import (
    HybridPanelIntegrator,
    Stokes3DHybrid,
    Stokes3DHybridMultigrid,
    edge_clustered_grid,
    hierarchical_grid,
    lattice_edge_grid,
)

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "benchmarks/legacy/benchmark_stokeslet_hybrid.py"
        ),
        run_name="__main__",
    )

__all__ = [
    "HybridPanelIntegrator",
    "Stokes3DHybrid",
    "Stokes3DHybridMultigrid",
    "edge_clustered_grid",
    "hierarchical_grid",
    "lattice_edge_grid",
]
