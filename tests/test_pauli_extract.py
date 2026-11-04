"""Simmons extraction checked against circuit matrices and direct MBQC sums."""

import itertools
import random
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np

import pyzx as zx

from pyzx import Circuit, Graph, simplify
from pyzx.circuit.gates import InitAncilla
from pyzx.flow import gflow, pauli_flow
from pyzx.pauli_extract import (
    CliffordMap, Measurement, PauliExtractionError, PauliPattern, PauliString,
    extract_circuit_pauli, extract_pauli,
)
from pyzx.symbolic import new_var
from pyzx.utils import EdgeType, VertexType


def pattern_matrix(pattern: PauliPattern) -> np.ndarray:
    """Independent positive-outcome state sum; wire zero is most significant.

    This uses measurement bras and graph-state amplitudes, without flow,
    stabilizer arithmetic, Pauli-string extraction or gate synthesis.
    """
    result = np.zeros((2 ** len(pattern.outputs), 2 ** len(pattern.inputs)), dtype=complex)
    for assignment in itertools.product((0, 1), repeat=len(pattern.vertices)):
        bits = dict(zip(pattern.vertices, assignment))
        amplitude = complex(2 ** (-(len(pattern.vertices) - len(pattern.inputs)) / 2))
        amplitude *= (-1) ** sum(bits[u] * bits[v] for u, v in pattern.edges)
        for v, measurement in pattern.measurements.items():
            angle = float(measurement.angle) * np.pi
            label = measurement.label
            if label == "XY":
                bra = np.array([1, np.exp(-1j * angle)]) / np.sqrt(2)
            elif label == "XZ":
                bra = np.array([np.cos(angle / 2), np.sin(angle / 2)])
            elif label == "YZ":
                bra = np.array([np.cos(angle / 2), -1j * np.sin(angle / 2)])
            elif label == "X":
                bra = np.array([1, (-1) ** int(measurement.angle)]) / np.sqrt(2)
            elif label == "Y":
                bra = np.array([1, -1j * (-1) ** int(measurement.angle)]) / np.sqrt(2)
            else:
                bra = np.array([1, 0] if measurement.angle == 0 else [0, 1])
            amplitude *= bra[bits[v]]
        row = sum(bits[v] << (len(pattern.outputs) - 1 - j) for j, v in enumerate(pattern.outputs))
        col = sum(bits[v] << (len(pattern.inputs) - 1 - j) for j, v in enumerate(pattern.inputs))
        result[row, col] += amplitude
    return result


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

    def test_all_six_labels_and_signed_outcomes(self):
        labels = ("XY", "XZ", "YZ", "X", "Y", "Z")
        vertices = (7, 31, 105)
        possible_edges = tuple(itertools.combinations(vertices, 2))
        checked = 0
        for edge_bits in range(8):
            edges = tuple(e for j, e in enumerate(possible_edges) if edge_bits & (1 << j))
            for a, b in itertools.product(labels, repeat=2):
                for signs in itertools.product((0, 1), repeat=2):
                    measurements = {
                        v: Measurement(label, Fraction(sign) if len(label) == 1 else Fraction(1 + 2 * sign, 3))
                        for v, label, sign in zip(vertices[:2], (a, b), signs)
                    }
                    pattern = PauliPattern(vertices, edges, (7,), (105,), measurements)
                    flow = pattern.find_flow()
                    if flow is None:
                        continue
                    with self.subTest(edges=edge_bits, labels=(a, b), signs=signs):
                        result = extract_pauli(pattern, flow=flow)
                        expected = pattern_matrix(pattern)
                        self.assert_same(expected, result.to_matrix())
                        self.assert_same(expected, result.to_circuit().to_matrix())
                    checked += 1
        self.assertEqual(checked, 180)

    def test_mixed_patterns_with_overlapping_boundaries(self):
        rng = random.Random(20261019)
        positives = 0
        for _ in range(240):
            vertices = tuple(range(rng.randrange(3, 7)))
            edges = tuple(e for e in itertools.combinations(vertices, 2) if rng.random() < 0.4)
            inputs = tuple(rng.sample(vertices, 2))
            outputs = tuple(rng.sample(vertices, 2))
            measurements = {}
            for v in set(vertices) - set(outputs):
                label = rng.choice(("XY", "XZ", "YZ", "X", "Y", "Z"))
                measurements[v] = Measurement(label, Fraction(rng.randrange(2)) if len(label) == 1 else Fraction(1, 5))
            pattern = PauliPattern(vertices, edges, inputs, outputs, measurements)
            flow = pattern.find_flow()
            if flow is not None:
                expected = pattern_matrix(pattern)
                result = extract_pauli(pattern, flow=flow)
                self.assert_same(expected, result.to_matrix())
                self.assert_same(expected, result.to_circuit().to_matrix())
                positives += 1
        self.assertGreaterEqual(positives, 20)

    def test_unbalanced_patterns_all_labels_against_state_sum(self):
        vertices = (7, 31, 105)
        possible_edges = tuple(itertools.combinations(vertices, 2))
        checked = set()
        for outputs in ((31, 105), (105, 7)):
            measured = next(v for v in vertices if v not in outputs)
            for edge_bits in range(8):
                edges = tuple(e for j, e in enumerate(possible_edges) if edge_bits & (1 << j))
                for label in ("XY", "XZ", "YZ", "X", "Y", "Z"):
                    for sign in (0, 1):
                        angle = Fraction(sign) if len(label) == 1 else Fraction(1 + 2 * sign, 3)
                        pattern = PauliPattern(vertices, edges, (7,), outputs,
                                               {measured: Measurement(label, angle)})
                        flow = pattern.find_flow()
                        if flow is None:
                            continue
                        with self.subTest(outputs=outputs, edges=edge_bits, label=label, sign=sign):
                            result = extract_pauli(pattern, flow=flow)
                            expected = pattern_matrix(pattern)
                            actual = result.to_matrix()
                            self.assert_same(expected, actual)
                            self.assert_same(expected, result.to_circuit().to_matrix())
                            self.assertEqual((result.n_inputs, result.n_outputs, result.n_qubits), (1, 2, 2))
                            self.assertEqual(len(result.clifford.stabilizers), 1)
                            np.testing.assert_allclose(actual.conj().T @ actual, np.eye(2), atol=1e-9)
                        checked.add(label)
        self.assertEqual(checked, {"XY", "XZ", "YZ", "X", "Y", "Z"})

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

    def test_unbalanced_flow_against_exhaustive_correction_search(self):
        rng = random.Random(20261105)
        positives = negatives = 0
        for case in range(120):
            vertices = tuple(range(3 + case % 3))
            edges = tuple(e for e in itertools.combinations(vertices, 2) if rng.random() < 0.45)
            inputs = tuple(rng.sample(vertices, case % 2))
            outputs = tuple(rng.sample(vertices, len(inputs) + 1 + case % 2))
            measurements = {v: Measurement(label, Fraction(rng.randrange(2)) if len(label) == 1 else Fraction(1, 5))
                            for v in vertices if v not in outputs
                            for label in (rng.choice(("XY", "XZ", "YZ", "X", "Y", "Z")),)}
            pattern = PauliPattern(vertices, edges, inputs, outputs, measurements)
            # Exhaust all correction subsets directly from the focused-flow
            # definition, then peel those with no remaining planar dependency.
            # This uses neither the leaf reduction nor binary elimination.
            correctors = tuple(v for v in vertices if v not in inputs)
            candidates = {v: [] for v in measurements}
            for bits in range(1 << len(correctors)):
                s = {v for j, v in enumerate(correctors) if bits & (1 << j)}
                odd = {v for v in vertices if sum(
                    (u, v) in edges or (v, u) in edges for u in s) % 2}
                for u, measurement in measurements.items():
                    own, own_odd = u in s, u in odd
                    local = {"XY": not own and own_odd, "XZ": own and own_odd,
                             "YZ": own and not own_odd, "X": own_odd,
                             "Y": own != own_odd, "Z": own}[measurement.label]
                    focused = all(
                        v not in odd if m.label in ("XY", "X") else
                        v not in s if m.label in ("XZ", "YZ", "Z") else
                        (v in s) == (v in odd)
                        for v, m in measurements.items() if v != u)
                    if local and focused:
                        candidates[u].append({v for v, m in measurements.items()
                                              if v != u and len(m.label) == 2 and v in s | odd})
            remaining = set(measurements)
            while remaining:
                ready = {u for u in remaining if any(not (deps & remaining) for deps in candidates[u])}
                if not ready:
                    break
                remaining -= ready
            flow = pattern.find_flow()
            with self.subTest(case=case):
                self.assertEqual(flow is not None, not remaining)
                if flow is not None:
                    result = extract_pauli(pattern, flow=flow)
                    self.assert_same(pattern_matrix(pattern), result.to_matrix())
                    self.assert_same(pattern_matrix(pattern), result.to_circuit().to_matrix())
                    positives += 1
                else:
                    negatives += 1
        self.assertGreater(positives, 20)
        self.assertGreater(negatives, 20)

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
        with patch("pyzx.pauli_extract.pauli_flow", side_effect=AssertionError("another solve")), \
             patch.object(PauliPattern, "find_flow", side_effect=AssertionError("another solve")):
            result = extract_pauli(graph, flow=flow)
        self.assert_same(graph.to_matrix(), result.to_matrix())
        self.assert_same(graph.to_matrix(), result.to_circuit().to_matrix())
        vertices = tuple(range(70))
        pattern = PauliPattern(vertices, (), (0,), tuple(reversed(vertices)), {})
        result = extract_pauli(pattern)
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
        with patch("pyzx.pauli_extract.pauli_flow", side_effect=AssertionError("another flow solve")), \
             patch.object(PauliPattern, "find_flow", side_effect=AssertionError("another flow solve")), \
             patch("pyzx.pauli_extract.np.eye", side_effect=AssertionError("dense arithmetic")):
            actual = extract_circuit_pauli(graph, flow=flow)
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
        vertices = tuple(17 + 3 * i for i in range(70))
        pattern = PauliPattern(vertices, (), vertices, tuple(reversed(vertices)), {})
        result = extract_pauli(pattern)
        self.assertEqual(result.clifford.x_images[0], PauliString(70, 1 << 69, 0))
        self.assertEqual(result.clifford.z_images[69], PauliString(70, 0, 1))
        self.assertEqual(result.to_circuit().qubits, 70)
        with self.assertRaisesRegex(ValueError, "dense verification"):
            result.to_matrix()

    def test_long_path_and_empty_pattern(self):
        pattern = PauliPattern(tuple(range(130)), tuple((v, v + 1) for v in range(129)),
                               (0,), (129,), {v: Measurement("X") for v in range(129)})
        circuit = Circuit(1)
        circuit.add_gate("HAD", 0)
        self.assert_same(circuit.to_matrix(), extract_circuit_pauli(pattern).to_circuit().to_matrix())
        empty = extract_pauli(PauliPattern((), (), (), (), {}))
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

    def test_corrupted_witnesses_are_rejected(self):
        pattern = PauliPattern((0, 1), ((0, 1),), (0,), (1,), {0: Measurement("XY", Fraction(1, 3))})
        flow = pattern.find_flow()
        assert flow is not None
        for bad in ((flow[0], {0: set()}), (flow[0], {0: {0}}),
                    ({0: 1, 1: 0}, flow[1]), ({0: 0}, flow[1]),
                    ({0: False, 1: 1}, flow[1])):
            with self.assertRaises(PauliExtractionError):
                extract_pauli(pattern, flow=bad)

    def test_domain_errors_and_exact_angles(self):
        tiny = Fraction(1, 2**60)
        circuit = Circuit(1)
        circuit.add_gate("ZPhase", 0, phase=tiny)
        graph = circuit.to_graph()
        simplify.to_graph_like(graph)
        self.assertTrue(any(r.angle in (tiny, 2 - tiny) for r in extract_pauli(graph).rotations))
        no_flow = PauliPattern((0, 1), (), (0,), (1,), {0: Measurement("XY")})
        with self.assertRaisesRegex(PauliExtractionError, "no Pauli flow"):
            extract_pauli(no_flow)
        with self.assertRaisesRegex(PauliExtractionError, "at least as many outputs"):
            extract_pauli(PauliPattern((0, 1), (), (0, 1), (1,), {0: Measurement("X")}))
        with self.assertRaisesRegex(PauliExtractionError, "axis measurement"):
            Measurement("Z", Fraction(1, 4))
        with self.assertRaisesRegex(PauliExtractionError, "unknown measurement"):
            Measurement("ZX")
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

    def test_validation_and_snapshotting(self):
        measurements = {0: Measurement("XY")}
        pattern = PauliPattern((0, 1), ((0, 1),), (0,), (1,), measurements)
        measurements.clear()
        self.assertIn(0, pattern.measurements)
        for kwargs in ({"vertices": (0, 0)}, {"edges": ((0, 0),)},
                       {"edges": ((0, 1), (1, 0))}, {"inputs": (0, 0)},
                       {"output_hadamards": (1,)}, {"output_phases": (Fraction(0), Fraction(0))}):
            with self.assertRaises(PauliExtractionError):
                replace(pattern, **kwargs)


if __name__ == "__main__":
    unittest.main()
