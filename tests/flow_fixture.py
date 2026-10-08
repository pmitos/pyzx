# Copyright (C) 2026 - Piotr Bartosz Mitosek
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Load the fixed flow example without external corpus dependencies."""

from fractions import Fraction
import json
from pathlib import Path

from pyzx.graph import Graph
from pyzx.utils import EdgeType, VertexType


FIXTURE_PATH = Path(__file__).parent / "data" / "flow" / "xy_x_y_n200_r00.json"


def load_flow_fixture():
    """Build the seed-0 graph: 200 spiders and 28 explicit boundaries."""
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    phases = {"XY": Fraction(1, 4), "X": Fraction(0),
              "Y": Fraction(1, 2), "OUTPUT": Fraction(1, 4)}
    graph = Graph(backend="simple")
    vertices = [graph.add_vertex(VertexType.Z, phase=phases[label])
                for label in data["labels"]]
    for u, v in data["edges"]:
        graph.add_edge((vertices[u], vertices[v]), EdgeType.HADAMARD)
    for side, setter in ((data["inputs"], graph.set_inputs),
                         (data["outputs"], graph.set_outputs)):
        boundaries = []
        for v in side:
            boundary = graph.add_vertex(VertexType.BOUNDARY)
            graph.add_edge((vertices[v], boundary), EdgeType.SIMPLE)
            boundaries.append(boundary)
        setter(tuple(boundaries))
    return graph
