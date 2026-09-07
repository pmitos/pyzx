# PyZX - Python library for quantum circuit rewriting
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import random
from fractions import Fraction

from pyzx.gflow import gflow
from pyzx.graph import Graph
from pyzx.utils import EdgeType, VertexType


class TestGFlow(unittest.TestCase):

    def assert_valid_flow(self, graph, result, pauli, reverse, focus):
        """Check correction parity and order independently of either solver."""
        self.assertIsNotNone(result)
        if result is None:
            return
        layers, corrections = result
        vertices = {v for v in graph.vertices() if graph.type(v) == VertexType.Z}
        inputs = {v for b in graph.inputs() for v in graph.neighbors(b)}
        outputs = {v for b in graph.outputs() for v in graph.neighbors(b)}
        if reverse:
            inputs, outputs = outputs, inputs
        grounds = graph.grounds()
        self.assertEqual(set(layers), vertices)
        self.assertEqual(set(corrections), vertices - outputs - grounds)
        for u, correction in corrections.items():
            self.assertFalse(correction & inputs)
            odd = {v for v in vertices
                   if len(set(graph.neighbors(v)) & correction) % 2}
            for v in vertices - outputs:
                if v in grounds and not focus:
                    continue
                phase = graph.phase(v) % 2
                is_y = pauli and phase in (Fraction(1, 2), Fraction(3, 2))
                is_x = pauli and phase in (0, 1)
                coefficient = (v in odd) ^ (is_y and v in correction)
                self.assertEqual(coefficient, v == u)
                if not is_x and not is_y and v in correction and v not in grounds:
                    self.assertNotEqual(u, v)
                    if reverse:
                        self.assertLess(layers[v], layers[u])
                    else:
                        self.assertGreater(layers[v], layers[u])

    def test_random_flow_equivalence(self):
        """Compare existence and validate cubic witnesses in every API mode."""
        rng = random.Random(506)
        phases = (Fraction(0), Fraction(1, 4), Fraction(1, 2),
                  Fraction(1), Fraction(3, 2))
        for sample in range(160):
            graph = Graph()
            vertices = [graph.add_vertex(VertexType.Z, phase=rng.choice(phases))
                        for _ in range(rng.randrange(1, 10))]
            for i, v in enumerate(vertices):
                for w in vertices[:i]:
                    if rng.random() < 0.35:
                        graph.add_edge((v, w), EdgeType.HADAMARD)
            boundaries = []
            for probability in (0.25, 0.4):
                side = []
                for v in vertices:
                    if rng.random() < probability:
                        b = graph.add_vertex(VertexType.BOUNDARY)
                        graph.add_edge((b, v), EdgeType.SIMPLE)
                        side.append(b)
                boundaries.append(tuple(side))
            graph.set_inputs(boundaries[0])
            graph.set_outputs(boundaries[1])
            for pauli in (False, True):
                for reverse in (False, True):
                    for focus in (False, True):
                        with self.subTest(sample=sample, pauli=pauli,
                                          reverse=reverse, focus=focus):
                            result = gflow(graph, focus, reverse, pauli)
                            legacy = gflow(graph, focus, reverse, pauli, method="legacy")
                            self.assertEqual(result is None, legacy is None)
                            if result is not None:
                                self.assert_valid_flow(graph, result, pauli, reverse, focus)

    def test_grounds(self):
        """Ground constraints follow PyZX's focus flag convention."""
        graph = Graph()
        u = graph.add_vertex(VertexType.Z, phase=Fraction(1, 4))
        ground = graph.add_vertex(VertexType.Z, ground=True)
        graph.add_edge((u, ground), EdgeType.HADAMARD)
        for focus in (False, True):
            result = gflow(graph, focus=focus)
            self.assertEqual(result is None, gflow(graph, focus=focus, method="legacy") is None)
            self.assert_valid_flow(graph, result, False, False, focus)

    def test_long_chain(self):
        """Columns unlocked in successive layers retain earlier constraints."""
        graph = Graph()
        vertices = [graph.add_vertex(VertexType.Z, phase=Fraction(1, 4))
                    for _ in range(40)]
        for u, v in zip(vertices, vertices[1:]):
            graph.add_edge((u, v), EdgeType.HADAMARD)
        for v, setter in ((vertices[0], graph.set_inputs),
                          (vertices[-1], graph.set_outputs)):
            b = graph.add_vertex(VertexType.BOUNDARY)
            graph.add_edge((v, b), EdgeType.SIMPLE)
            setter((b,))
        self.assert_valid_flow(graph, gflow(graph), False, False, False)

    def test_empty_and_isolated(self):
        """Empty graphs and diagonal-only Y corrections need no graph edges."""
        graph = Graph()
        self.assertEqual(gflow(graph), ({}, {}))
        v = graph.add_vertex(VertexType.Z, phase=Fraction(1, 2))
        self.assertIsNone(gflow(graph))
        self.assertEqual(gflow(graph, pauli=True), ({v: 0}, {v: {v}}))
        with self.assertRaises(ValueError):
            gflow(graph, method="unknown")

    def test_pauli_y_diagonal_correction(self):
        """A Pauli-Y vertex can use the diagonal of the demand matrix."""
        graph = Graph()
        xy = graph.add_vertex(VertexType.Z, 0, 1, phase=Fraction(1, 4))
        output = graph.add_vertex(VertexType.Z, 1, 1)
        y = graph.add_vertex(VertexType.Z, 2, 1, phase=Fraction(1, 2))
        graph.add_edge((xy, output), EdgeType.HADAMARD)
        graph.add_edge((xy, y), EdgeType.HADAMARD)

        input_boundary = graph.add_vertex(VertexType.BOUNDARY, 0, 0)
        output_boundary = graph.add_vertex(VertexType.BOUNDARY, 1, 2)
        graph.add_edge((input_boundary, xy), EdgeType.SIMPLE)
        graph.add_edge((output, output_boundary), EdgeType.SIMPLE)
        graph.set_inputs((input_boundary,))
        graph.set_outputs((output_boundary,))

        for focus in (False, True):
            with self.subTest(focus=focus):
                result = gflow(graph, focus=focus, pauli=True)
                self.assertIsNotNone(result)
                if result is None:
                    continue
                _, corrections = result
                self.assertEqual(corrections[xy], {output})
                self.assertEqual(corrections[y], {output, y})


if __name__ == '__main__':
    unittest.main()
