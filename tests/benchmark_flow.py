# Copyright (C) 2026 - Piotr Bartosz Mitosek
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Opt-in full-witness comparison: python -m tests.benchmark_flow."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter, process_time
from typing import Any

from pyzx import flow
from pyzx.graph import Graph
from pyzx.utils import EdgeType, VertexType
from .flow_fixture import FIXTURE_PATH, load_flow_fixture
from .test_gflow import TestGFlow


def warm_up():
    """Exercise both finders on a tiny positive graph outside timing."""
    graph = Graph()
    xy = graph.add_vertex(VertexType.Z, phase=Fraction(1, 4))
    output = graph.add_vertex(VertexType.Z, phase=Fraction(1, 4))
    graph.add_vertex(VertexType.Z, phase=Fraction(1, 2))
    boundary = graph.add_vertex(VertexType.BOUNDARY)
    graph.add_edge((xy, output), EdgeType.HADAMARD)
    graph.add_edge((output, boundary), EdgeType.SIMPLE)
    graph.set_outputs((boundary,))
    checker = TestGFlow()
    for method in ("incremental", "legacy"):
        result = flow.pauli_flow(graph, focus=True, method=method)
        checker.assert_valid_flow(graph, result, True, False, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Save timings and provenance as JSON")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    finder_path = Path(flow.__file__)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=finder_path.parent.parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report: dict[str, Any] = {
        "case_id": fixture["case_id"], "spiders": fixture["n"],
        "boundary_vertices": len(fixture["inputs"]) + len(fixture["outputs"]),
        "fixture_sha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        "finder_sha256": hashlib.sha256(finder_path.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(Path(__file__).with_name("flow_fixture.py").read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": commit, "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "focus": True, "layers_only": False, "repeats": args.repeats,
        "method_order": "incremental first on even repetitions, legacy first on odd repetitions",
        "records": [],
    }
    print(f"{fixture['case_id']}: {fixture['n']} spiders, full corrections, focus=True", flush=True)
    print("Timing finder calls only; graph loading, warm-up, and witness checks are excluded.", flush=True)
    warm_up()
    checker = TestGFlow()
    times: dict[str, list[float]] = {method: [] for method in ("incremental", "legacy")}
    cpu_times: dict[str, list[float]] = {method: [] for method in times}
    for repetition in range(args.repeats):
        methods = ("incremental", "legacy") if repetition % 2 == 0 else ("legacy", "incremental")
        for method in methods:
            graph = load_flow_fixture()
            before = graph.to_json()
            cpu_start = process_time()
            start = perf_counter()
            result = flow.pauli_flow(graph, focus=True, method=method)
            elapsed = perf_counter() - start
            cpu_elapsed = process_time() - cpu_start
            checker.assert_valid_flow(graph, result, True, False, True)
            checker.assertEqual(graph.to_json(), before)
            times[method].append(elapsed)
            cpu_times[method].append(cpu_elapsed)
            report["records"].append({
                "repetition": repetition, "method": method,
                "finder_seconds": elapsed, "finder_cpu_seconds": cpu_elapsed,
                "witness_valid": True, "graph_unchanged": True,
            })
            print(f"{method:11s} repetition {repetition + 1}: {elapsed:.6f} s", flush=True)
    report["median_seconds"] = {method: statistics.median(values) for method, values in times.items()}
    report["median_cpu_seconds"] = {method: statistics.median(values) for method, values in cpu_times.items()}
    medians = report["median_seconds"]
    report["legacy_over_incremental"] = medians["legacy"] / medians["incremental"]
    print(f"Median legacy / incremental: {report['legacy_over_incremental']:.1f}x")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
