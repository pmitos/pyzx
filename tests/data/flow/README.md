# Fixed Pauli-flow example

`xy_x_y_n200_r00.json` is a standalone copy of the positive
`XY_X_Y/n00200-r00` example from
[UnlabelledOpenGraphs](https://github.com/pmitos/UnlabelledOpenGraphs/tree/167a9dea83a270b9e4789b2229fb7ace4392357e/mbqcflow/tests/data/flow_benchmarks/XY_X_Y/n00200-r00).
The source repository and this fixture use the Apache License, Version 2.0.
No mbqcflow, NetworkX, or NumPy loader is needed to read this JSON fixture.

## Provenance and representation

- Source commit: `167a9dea83a270b9e4789b2229fb7ace4392357e`.
- Source NPZ SHA-256:
  `c7915ec99afa45c143f2276433fc5ff9a554ca015659837ddf63e708b6d7a2bc`.
- Fixture SHA-256:
  `cc719cd4e9f238eb9cce55d170a399112e6b4e7df252e2c2029c30b4113ad14e`.
- 200 spiders, 1,655 undirected graph edges, 14 inputs, 14 outputs.
- Measured labels: 169 XY, 15 X, and 2 Y; the other 14 vertices are outputs.

The fixture preserves the archived seed-0 ordering. `source.vertex_order`
maps fixture IDs to archive IDs. Archive IDs are sorted by SHA-256 of
`sha256-vertex-order-v1:XY_X_Y:200:0:0:VERTEX`, with the archive ID as the
tie-breaker. Edges, labels, inputs, and outputs are all remapped together.
The JSON edges are sorted undirected pairs without duplicates.

The loader uses Z spiders with phases 1/4 for XY, 0 for X, and 1/2 for Y.
Outputs use phase 1/4 but are unmeasured. Graph edges are Hadamard edges.
Each input and output gets an explicit boundary connected by a simple edge,
giving 228 PyZX vertices and 1,683 total edges. This matches the archived
PyZX conversion. Pauli signs are irrelevant to flow existence here.
The example has Pauli flow and has no flow when all measured vertices are XY.

## Correctness and optional timing

The ordinary test checks the incremental witness independently, checks exact
layers-only agreement, and checks the all-XY negative variant. It has no
timing assertion and does not call legacy on this fixture.

From the PyZX repository root:

```sh
python -m tests.benchmark_flow --repeats 3 --output flow-benchmark.json
```

The command compares the current full incremental and legacy finders with
`focus=True`. Each timed call receives a freshly constructed copy of the same
fixture. Call order alternates by repetition. A tiny positive graph warms both
methods outside timing. The existing independent witness checker in
`tests/test_gflow.py` checks every timed result, and mutation checks follow it.
Graph loading, conversion, warm-up, and validation are outside finder timing.
Both methods return correction sets; layers-only mode is not timed here.

The report saves individual wall/CPU times, their medians, the ratio of wall
medians, fixture/finder/loader/benchmark SHA-256 hashes, Git commit, Python,
and platform information. The source hashes identify the actual files even
when running a modified working tree. There is no assumed speedup or passing
threshold. This is one deliberately structured positive case, not a claim
about the distribution of all open graphs.

For historical context, the September LUH campaign at frozen PyZX `ca58dbe`
recorded five successful repetitions per method for this case: median legacy
30.465013 s and incremental 0.008215168 s. These archived measurements are not
timings of the current code and are not used as test expectations. See the
[campaign report](https://github.com/pmitos/UnlabelledOpenGraphs/blob/167a9dea83a270b9e4789b2229fb7ace4392357e/mbqcflow/benchmarks/luh-2026-09-16/BENCHMARK_FINAL_2026-09-21.md)
and its linked raw records for the frozen environment and protocol.
