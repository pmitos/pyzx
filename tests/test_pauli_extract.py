"""Simmons extraction checked against circuit and ZX-diagram matrices."""

import itertools
import random
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np

import pyzx as zx

from pyzx import Circuit, Graph, simplify
from pyzx.circuit.gates import InitAncilla, Reset
from pyzx.flow import gflow, pauli_flow
from pyzx.graph.base import BaseGraph
from pyzx.pauli_extract import (
    CliffordMap, PauliExtractionError, PauliString,
    extract_circuit_pauli, extract_pauli,
)
from pyzx.symbolic import new_var
from pyzx.utils import EdgeType, VertexType


def open_graph(
    phases: tuple[Fraction | int, ...], edges: tuple[tuple[int, int], ...],
    inputs: tuple[int, ...], outputs: tuple[int, ...],
) -> BaseGraph[int, tuple[int, int]]:
    """Build a graph-like ZX diagram from spiders and boundary positions."""
    g = Graph()
    spiders = [g.add_vertex(VertexType.Z, phase=phase) for phase in phases]
    for u, v in edges:
        g.add_edge((spiders[u], spiders[v]), EdgeType.HADAMARD)
    boundaries = []
    for wires in (inputs, outputs):
        boundary = []
        for i in wires:
            b = g.add_vertex(VertexType.BOUNDARY)
            g.add_edge((b, spiders[i]))
            boundary.append(b)
        boundaries.append(tuple(boundary))
    g.set_inputs(boundaries[0])
    g.set_outputs(boundaries[1])
    return g


class TestPauliExtract(unittest.TestCase):
    def assert_same(self, expected: np.ndarray, actual: np.ndarray) -> None:
        self.assertEqual(expected.shape, actual.shape)
        index = np.unravel_index(np.abs(expected).argmax(), expected.shape)
        self.assertGreater(abs(expected[index]), 1e-10)
        self.assertGreater(abs(actual[index]), 1e-10)
        np.testing.assert_allclose(actual / actual[index], expected / expected[index],
                                   atol=1e-9, rtol=0)

    def test_random_circuits_and_rewritten_diagrams(self):
        rng = random.Random(20261019)
        for case in range(20):
            qubits = 1 + case % 4
            circuit = Circuit(qubits)
            for _ in range(15):
                choices = ("HAD", "S", "T", "ZPhase", "CNOT", "CZ", "SWAP") if qubits > 1 else ("HAD", "S", "T", "ZPhase")
                name = rng.choice(choices)
                if name in ("CNOT", "CZ", "SWAP"):
                    circuit.add_gate(name, *rng.sample(range(qubits), 2))
                elif name == "ZPhase":
                    circuit.add_gate(name, rng.randrange(qubits), phase=Fraction(rng.randrange(-5, 6), 5))
                else:
                    circuit.add_gate(name, rng.randrange(qubits))
            expected = circuit.to_matrix()
            for mode in ("graphlike", "clifford", "full"):
                with self.subTest(case=case, mode=mode):
                    graph = circuit.to_graph()
                    simplify.to_graph_like(graph)
                    if mode == "clifford":
                        simplify.clifford_simp(graph, quiet=True)
                    elif mode == "full":
                        simplify.full_reduce(graph, quiet=True)
                    original = graph.to_json()
                    flow = pauli_flow(graph)
                    self.assertIsNotNone(flow)
                    result = extract_pauli(graph, flow=flow)
                    self.assert_same(expected, result.to_matrix())
                    self.assert_same(expected, result.to_circuit().to_matrix())
                    self.assertEqual(graph.to_json(), original)

    def test_negative_pauli_input_measurements(self):
        for phase, x_image in ((Fraction(1), "-Z"), (Fraction(1, 2), "-Y")):
            with self.subTest(phase=phase):
                graph = open_graph((phase, 0), ((0, 1),), (0,), (1,))
                result = extract_pauli(graph)
                self.assertEqual(tuple(map(str, result.clifford.x_images)), (x_image,))
                self.assertEqual(tuple(map(str, result.clifford.z_images)), ("+X",))
                self.assertEqual(result.rotations, ())
                self.assert_same(graph.to_matrix(), result.to_matrix())

    def test_random_unbalanced_circuits_and_state_preparations(self):
        rng = random.Random(20261104)
        for case in range(12):
            inputs, extra = case % 3, 1 + case % 2
            outputs = inputs + extra
            circuit = Circuit(inputs)
            for q in range(inputs, outputs):
                circuit.add_gate(InitAncilla(q, rng.choice(("+", "-", "0", "1"))))
            for _ in range(12):
                name = rng.choice(("HAD", "S", "T", "ZPhase", "CNOT", "CZ", "SWAP")
                                  if outputs > 1 else ("HAD", "S", "T", "ZPhase"))
                if name in ("CNOT", "CZ", "SWAP"):
                    circuit.add_gate(name, *rng.sample(range(outputs), 2))
                elif name == "ZPhase":
                    circuit.add_gate(name, rng.randrange(outputs), phase=Fraction(rng.randrange(-5, 6), 5))
                else:
                    circuit.add_gate(name, rng.randrange(outputs))
            expected = circuit.to_matrix()
            for reduce in (False, True):
                with self.subTest(case=case, reduce=reduce):
                    graph = circuit.to_graph()
                    simplify.to_graph_like(graph)
                    if reduce:
                        simplify.full_reduce(graph, quiet=True)
                    # Output order is part of the isometry, not a canonical wire order.
                    graph.set_outputs(tuple(reversed(graph.outputs())))
                    permuted = graph.to_matrix()
                    original = graph.to_json()
                    for extractor in (extract_pauli, extract_circuit_pauli):
                        with patch.object(CliffordMap, "to_circuit", side_effect=AssertionError("synthesis")), \
                             patch("pyzx.pauli_extract.np.eye", side_effect=AssertionError("dense arithmetic")):
                            result = extractor(graph)
                        self.assertEqual((result.n_inputs, result.n_outputs), (inputs, outputs))
                        self.assertEqual(len(result.clifford.stabilizers), extra)
                        self.assert_same(permuted, result.to_matrix())
                        gates = result.to_circuit()
                        self.assertEqual(gates.qubits, inputs)
                        self.assertEqual([g.label for g in gates.gates if isinstance(g, InitAncilla)],
                                         list(range(inputs, outputs)))
                        self.assertTrue(all(g.state == "+" for g in gates.gates if isinstance(g, InitAncilla)))
                        self.assert_same(permuted, gates.to_matrix())
                        self.assertEqual(graph.to_json(), original)
                    # Undo the permutation to compare with the source circuit too.
                    graph.set_outputs(tuple(reversed(graph.outputs())))
                    self.assert_same(expected, extract_pauli(graph).to_matrix())

    def test_isometry_tableau_specifies_logical_action_and_output_subspace(self):
        # A signed Bell state has no input images; the stabilizers are essential.
        bell = CliffordMap((), (), (PauliString(2, 3, 0, -1), PauliString(2, 0, 3)))
        self.assert_same(np.array([[1], [0], [0], [-1]]), bell.to_matrix())
        self.assert_same(bell.to_matrix(), bell.to_circuit().to_matrix())
        # Encode |0> -> |00>, |1> -> |11>, with a negative logical-X image.
        code = CliffordMap((PauliString(2, 3, 0, -1),), (PauliString(2, 0, 1),),
                           (PauliString(2, 0, 3),))
        matrix = code.to_matrix()
        self.assert_same(np.array([[1, 0], [0, 0], [0, 0], [0, -1]]), matrix)
        self.assert_same(matrix, code.to_circuit().to_matrix())
        for image, source in ((code.x_images[0], PauliString(1, 1, 0)),
                              (code.z_images[0], PauliString(1, 0, 1))):
            np.testing.assert_allclose(image.to_matrix() @ matrix, matrix @ source.to_matrix(), atol=1e-9)
        np.testing.assert_allclose(code.stabilizers[0].to_matrix() @ matrix, matrix, atol=1e-9)
        for bad in (
            ((), (), (PauliString(1, 0, 0),)),
            ((), (), (PauliString(2, 1, 0), PauliString(2, 1, 0, -1))),
            ((), (), (PauliString(2, 1, 0), PauliString(2, 0, 1))),
            ((PauliString(2, 1, 0),), (PauliString(2, 0, 1),), (PauliString(2, 1, 0),)),
        ):
            with self.subTest(bad=bad), self.assertRaises(PauliExtractionError):
                CliffordMap(*bad)

    def test_isometry_qasm_export_preserves_preparation_and_rotations(self):
        # A graph with an X root and an XY leaf implements a YZ effect.
        graph = open_graph((0, 0, 0, Fraction(1, 3)), ((0, 1), (1, 2), (1, 3)), (0,), (0, 2))
        result = extract_pauli(graph)
        plus = np.array([[1], [1]]) / np.sqrt(2)
        phases = np.diag(np.exp(-1j * np.pi / 6 * np.array([1, -1, -1, 1])))
        expected = phases @ np.kron(np.eye(2), plus)
        self.assert_same(expected, graph.to_matrix())
        self.assert_same(expected, result.to_matrix())
        self.assert_same(expected, result.to_circuit().to_matrix())
        self.assertEqual([(str(r.pauli), r.angle) for r in result.rotations],
                         [("+ZZ", Fraction(1, 3))])
        for version in (2, 3):
            with self.subTest(version=version):
                imported = Circuit.from_qasm(result.to_circuit().to_qasm(version))
                self.assertEqual(imported.qubits, 2)
                self.assertEqual([g.target for g in imported.gates if isinstance(g, Reset)], [1])
                # Fix the extra input to zero and simplify its leading reset.
                imported.initialize_qubits([False, True])
                graph = imported.to_graph(elide_initial_resets=True)
                self.assert_same(expected, graph.to_matrix())

    def test_unbalanced_supplied_witness_and_large_packed_stabilizers(self):
        graph = Graph()
        inp = graph.add_vertex(VertexType.BOUNDARY)
        out = graph.add_vertex(VertexType.BOUNDARY)
        graph.add_edge((inp, out), EdgeType.HADAMARD)
        state = graph.add_vertex(VertexType.Z, phase=Fraction(1, 7))
        state_out = graph.add_vertex(VertexType.BOUNDARY)
        graph.add_edge((state, state_out), EdgeType.HADAMARD)
        graph.set_inputs((inp,))
        graph.set_outputs((state_out, out))
        flow = pauli_flow(graph)
        with patch("pyzx.pauli_extract.pauli_flow", side_effect=AssertionError("another solve")):
            result = extract_pauli(graph, flow=flow)
        self.assert_same(graph.to_matrix(), result.to_matrix())
        self.assert_same(graph.to_matrix(), result.to_circuit().to_matrix())
        vertices = tuple(range(70))
        graph = open_graph((0,) * 70, (), (0,), tuple(reversed(vertices)))
        result = extract_pauli(graph)
        self.assertEqual(len(result.clifford.stabilizers), 69)
        self.assertEqual(result.clifford.x_images[0], PauliString(70, 1 << 69, 0))
        self.assertEqual(result.to_circuit().qubits, 1)
        self.assertEqual(len(result.to_circuit().to_graph().outputs()), 70)
        with self.assertRaisesRegex(ValueError, "dense verification"):
            result.to_matrix()

    def test_shared_output_spiders_and_witness_reuse(self):
        for inputs, mask, infer in itertools.product((0, 1), range(4), (False, True)):
            graph = Graph()
            root = graph.add_vertex(VertexType.Z, phase=Fraction(1, 3))
            ins, outs = [], []
            for _ in range(inputs):
                b = graph.add_vertex(VertexType.BOUNDARY)
                graph.add_edge((b, root), EdgeType.HADAMARD if mask & 1 else EdgeType.SIMPLE)
                ins.append(b)
            for q in range(3):
                b = graph.add_vertex(VertexType.BOUNDARY)
                graph.add_edge((root, b), EdgeType.HADAMARD if mask & (1 << q) else EdgeType.SIMPLE)
                outs.append(b)
            graph.set_inputs(tuple(ins))
            graph.set_outputs(tuple(reversed(outs)))
            finder = pauli_flow if infer else gflow
            flow = finder(graph)
            assert flow is not None
            saved = (dict(flow[0]), {v: set(s) for v, s in flow[1].items()})
            original = graph.to_json()
            expected = graph.to_matrix()
            with self.subTest(inputs=inputs, mask=mask, infer=infer):
                with patch("pyzx.pauli_extract.pauli_flow", side_effect=AssertionError("another solve")), \
                     patch("pyzx.pauli_extract.gflow", side_effect=AssertionError("another solve")):
                    result = extract_pauli(graph, flow=flow, infer_pauli=infer)
                self.assert_same(expected, result.to_matrix())
                self.assert_same(expected, result.to_circuit().to_matrix())
                self.assert_same(expected, extract_pauli(graph, infer_pauli=infer).to_matrix())
                self.assertEqual(flow, saved)
                self.assertEqual(graph.to_json(), original)
                with self.assertRaises(PauliExtractionError):
                    extract_pauli(graph, flow=({root: False}, {}), infer_pauli=infer)
                with self.assertRaises(PauliExtractionError):
                    extract_pauli(graph, flow=({root: 0, 1000: 0}, {}), infer_pauli=infer)

    def test_public_extractors_preserve_pauli_representation(self):
        circuit = Circuit(2)
        circuit.add_gate("HAD", 0)
        circuit.add_gate("CNOT", 0, 1)
        circuit.add_gate("ZPhase", 1, phase=Fraction(1, 7))
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        original = graph.to_json()
        witness = pauli_flow(graph)
        self.assertIsNotNone(witness)
        for extractor in (zx.extract_pauli, zx.extract_circuit_pauli):
            for flow in (None, witness):
                with self.subTest(extractor=extractor.__name__, supplied=flow is not None):
                    with patch.object(zx.PauliCircuit, "to_circuit",
                                      side_effect=AssertionError("implicit gate synthesis")), \
                         patch.object(CliffordMap, "to_circuit",
                                      side_effect=AssertionError("implicit Clifford synthesis")), \
                         patch("pyzx.pauli_extract.np.eye",
                               side_effect=AssertionError("dense arithmetic")):
                        result = extractor(graph, flow=flow)
                    self.assertIsInstance(result, zx.PauliCircuit)
                    self.assertTrue(result.rotations)
                    self.assertTrue(any(r.angle.denominator == 7 for r in result.rotations))
                    self.assert_same(circuit.to_matrix(), result.to_matrix())
                    gates = result.to_circuit()
                    self.assertIsInstance(gates, Circuit)
                    self.assert_same(circuit.to_matrix(), gates.to_matrix())
                    self.assertEqual(graph.to_json(), original)

    def test_supplied_public_witness_is_reused_without_dense_arithmetic(self):
        circuit = Circuit(2)
        circuit.add_gate("HAD", 0)
        circuit.add_gate("CNOT", 0, 1)
        circuit.add_gate("T", 1)
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        flow = pauli_flow(graph)
        with patch("pyzx.pauli_extract.pauli_flow", wraps=pauli_flow) as finder:
            found_inside = extract_circuit_pauli(graph, flow=None)
        finder.assert_called_once_with(graph)
        with patch("pyzx.pauli_extract.pauli_flow", side_effect=AssertionError("another flow solve")), \
             patch("pyzx.pauli_extract.np.eye", side_effect=AssertionError("dense arithmetic")):
            actual = extract_circuit_pauli(graph, flow=flow)
        self.assertEqual(actual, found_inside)
        self.assertIsInstance(actual, zx.PauliCircuit)
        self.assert_same(circuit.to_matrix(), actual.to_matrix())
        self.assert_same(circuit.to_matrix(), actual.to_circuit().to_matrix())

    def test_gflow_witness_and_no_axis_inference(self):
        circuit = Circuit(1)
        circuit.add_gate("S", 0)
        circuit.add_gate("HAD", 0)
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        self.assert_same(circuit.to_matrix(), extract_circuit_pauli(
            graph, flow=gflow(graph), infer_pauli=False).to_circuit().to_matrix())

    def test_boundaries_phases_and_direct_wires(self):
        circuit = Circuit(3)
        circuit.add_gate("HAD", 0)
        circuit.add_gate("ZPhase", 0, phase=Fraction(1, 7))
        circuit.add_gate("S", 2)
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        ins, outs = tuple(reversed(graph.inputs())), tuple(reversed(graph.outputs()))
        graph.set_inputs(ins)
        graph.set_outputs(outs)
        expected = graph.to_matrix()
        self.assert_same(expected, extract_circuit_pauli(graph).to_circuit().to_matrix())
        self.assert_same(expected, extract_circuit_pauli(
            graph, flow=pauli_flow(graph)).to_circuit().to_matrix())

    def test_large_packed_boundaries_without_dense_conversion(self):
        graph = open_graph((0,) * 70, (), tuple(range(70)), tuple(reversed(range(70))))
        result = extract_pauli(graph)
        self.assertEqual(result.clifford.x_images[0], PauliString(70, 1 << 69, 0))
        self.assertEqual(result.clifford.z_images[69], PauliString(70, 0, 1))
        self.assertEqual(result.to_circuit().qubits, 70)
        with self.assertRaisesRegex(ValueError, "dense verification"):
            result.to_matrix()

    def test_long_path_and_empty_graph(self):
        graph = open_graph((0,) * 130, tuple((v, v + 1) for v in range(129)), (0,), (129,))
        circuit = Circuit(1)
        circuit.add_gate("HAD", 0)
        self.assert_same(circuit.to_matrix(), extract_circuit_pauli(graph).to_circuit().to_matrix())
        empty = extract_pauli(Graph())
        np.testing.assert_array_equal(empty.to_matrix(), [[1]])
        self.assertEqual(empty.to_circuit().qubits, 0)
        self.assertEqual(empty.to_circuit().gates, [])

    def test_pauli_flow_without_xy_gflow(self):
        # A Y measurement at the middle of a triangle absorbs the cyclic
        # dependency which would obstruct an all-XY flow.
        for edges in itertools.combinations(tuple(itertools.combinations(range(4), 2)), 4):
            graph = Graph()
            vertices = [graph.add_vertex(VertexType.Z) for _ in range(4)]
            for u, v in edges:
                graph.add_edge((vertices[u], vertices[v]), EdgeType.HADAMARD)
            inp = graph.add_vertex(VertexType.BOUNDARY)
            out = graph.add_vertex(VertexType.BOUNDARY)
            graph.add_edge((inp, vertices[0]))
            graph.add_edge((out, vertices[3]))
            graph.set_inputs((inp,))
            graph.set_outputs((out,))
            graph.set_phase(vertices[1], Fraction(1, 2))
            graph.set_phase(vertices[2], Fraction(1, 3))
            flow = pauli_flow(graph)
            if flow is not None and gflow(graph) is None:
                self.assert_same(graph.to_matrix(), extract_circuit_pauli(
                    graph, flow=flow).to_circuit().to_matrix())
                return
        self.fail("no Pauli-only example found")

    def test_structurally_invalid_witnesses_are_rejected(self):
        graph = open_graph((Fraction(1, 3), 0), ((0, 1),), (0,), (1,))
        flow = pauli_flow(graph)
        assert flow is not None
        for bad in ((flow[0], {0: set()}), (flow[0], {0: {0}}),
                    ({0: 1, 1: 0}, flow[1]), ({0: 0}, flow[1]),
                    ({0: False, 1: 1}, flow[1])):
            with self.assertRaises(PauliExtractionError):
                extract_pauli(graph, flow=bad)

    def test_domain_errors_and_exact_angles(self):
        tiny = Fraction(1, 2**60)
        circuit = Circuit(1)
        circuit.add_gate("ZPhase", 0, phase=tiny)
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        self.assertTrue(any(r.angle in (tiny, 2 - tiny) for r in extract_pauli(graph).rotations))
        no_flow = open_graph((0, 0), (), (0,), (1,))
        with self.assertRaisesRegex(PauliExtractionError, "no Pauli flow"):
            extract_pauli(no_flow)
        with self.assertRaisesRegex(PauliExtractionError, "at least as many outputs"):
            extract_pauli(open_graph((0, 0), (), (0, 1), (1,)))
        with self.assertRaisesRegex(PauliExtractionError, "commutation"):
            CliffordMap((PauliString(1, 1, 0),), (PauliString(1, 1, 0),))
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        spider = next(v for v in graph.vertices() if graph.type(v) == VertexType.Z)
        graph.set_phase(spider, new_var("a", False))
        with self.assertRaisesRegex(PauliExtractionError, "non-symbolic"):
            extract_pauli(graph)
        graph.set_phase(spider, 0)
        graph.set_ground(spider, True)
        with self.assertRaisesRegex(PauliExtractionError, "grounds"):
            extract_pauli(graph)

    def test_input_spiders_must_be_distinct(self):
        graph = open_graph((0, 0), (), (0, 0), (0, 1))
        with self.assertRaisesRegex(PauliExtractionError, "inputs must be distinct"):
            extract_pauli(graph)


if __name__ == "__main__":
    unittest.main()
