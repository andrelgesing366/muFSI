# Research workspace

Use this folder for experimental quadratures, alternative panel algorithms,
operator/compression studies, and profiling scripts. It is not installed with
`mufsi`, and production modules must not import from it.

When promoting an experiment into the package, document its formulation and
scope, add independently meaningful validation, and move the reusable code to
the appropriate library module. Keep measured runtime/memory studies under
`benchmarks/` and reproducible user workflows under `examples/`.
