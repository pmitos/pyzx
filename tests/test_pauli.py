import unittest
from fractions import Fraction

import numpy as np

from pyzx import Circuit, Graph, PauliExponential, VertexType, tcount
from pyzx.circuit.gates import HAD, PhaseGadget, S, TargetMapper
from pyzx.symbolic import new_var


PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def exponential(axes: str, angle: Fraction | int) -> np.ndarray:
    pauli = np.array([[1]])
    for axis in axes:
        pauli = np.kron(pauli, PAULIS[axis])
    theta = np.pi * float(angle) / 2
    return np.cos(theta) * np.eye(len(pauli)) - 1j * np.sin(theta) * pauli


class TestPauliExponential(unittest.TestCase):
    def test_exact_graph_matrix(self):
        cases = (
            ("Y", "Y", Fraction(-1, 5), 1),
            ("+ZZ", "ZZ", Fraction(1, 3), 1),
            ("-XIYZ", "XIYZ", Fraction(1, 3), -1),
            ("XX", "XX", Fraction(5, 2), 1),
            ("-II", "II", Fraction(1, 4), -1),
            ("XZ", "XZ", 0, 1),
        )
        for pauli, axes, angle, sign in cases:
            with self.subTest(pauli=pauli, angle=angle):
                graph = PauliExponential(pauli, angle).to_graph()
                self.assertEqual(len(graph.inputs()), len(axes))
                self.assertEqual(len(graph.outputs()), len(axes))
                np.testing.assert_allclose(graph.to_matrix(), exponential(axes, sign * angle), atol=1e-12)

    def test_append_to_graph_with_wire_mapping(self):
        angle = Fraction(1, 3)
        for pauli, physical_axes in (("-XIY", "IYX"), ("-III", "III")):
            with self.subTest(pauli=pauli):
                graph = Graph()
                q_mapper, c_mapper = TargetMapper(), TargetMapper()
                inputs = [graph.add_vertex(VertexType.BOUNDARY, q, 0) for q in range(3)]
                graph.set_inputs(tuple(inputs))
                for label, qubit in enumerate((2, 0, 1)):
                    q_mapper.add_label(label, 1)
                    q_mapper.set_qubit(label, qubit)
                    q_mapper.set_prev_vertex(label, inputs[qubit])

                HAD(0).to_graph(graph, q_mapper, c_mapper)
                graph.scalar.add_phase(Fraction(1, 5))
                sites_before = tuple(q_mapper.prev_vertex(label) for label in range(3))
                PauliExponential(pauli, angle).to_graph(graph, q_mapper, c_mapper)
                for label, axis in enumerate(pauli[1:]):
                    if axis == "I":
                        self.assertEqual(q_mapper.prev_vertex(label), sites_before[label])
                    else:
                        self.assertNotEqual(q_mapper.prev_vertex(label), sites_before[label])
                S(2).to_graph(graph, q_mapper, c_mapper)

                outputs = [graph.add_vertex(VertexType.BOUNDARY, q, q_mapper.max_row())
                           for q in range(3)]
                for label in range(3):
                    graph.add_edge((q_mapper.prev_vertex(label), outputs[q_mapper.to_qubit(label)]))
                graph.set_outputs(tuple(outputs))

                before = np.kron(np.eye(4), np.array([[1, 1], [1, -1]]) / np.sqrt(2))
                after = np.kron(np.eye(2), np.kron(np.diag([1, 1j]), np.eye(2)))
                expected = np.exp(1j * np.pi / 5) * after @ exponential(physical_axes, -angle) @ before
                np.testing.assert_allclose(graph.to_matrix(), expected, atol=1e-12)

    def test_circuit_gate_and_tcount(self):
        angle = Fraction(1, 4)
        circuit = Circuit(3)
        circuit.add_gate("PauliExponential", "-XIY", angle)
        circuit.add_gate(PauliExponential("ZZ", Fraction(1, 2)))
        circuit.add_gate(PauliExponential("III", angle))
        circuit.add_gate(PauliExponential("IIX", 2))
        expected = (-np.exp(-1j * np.pi / 8)
                    * (np.kron(exponential("ZZ", Fraction(1, 2)), np.eye(2))
                       @ exponential("XIY", -angle)))
        np.testing.assert_allclose(circuit.to_matrix(), expected, atol=1e-12)
        np.testing.assert_allclose(circuit.adjoint().to_matrix(), expected.conj().T, atol=1e-12)
        self.assertEqual(circuit.copy().gates, circuit.gates)
        self.assertEqual(circuit.tcount(), 1)
        self.assertEqual(circuit.stats_dict(depth=True)["tcount"], 1)
        self.assertEqual(circuit.twoqubitcount(), 4)
        self.assertEqual(tcount(circuit.to_graph()), 1)
        self.assertEqual(PauliExponential("XX", Fraction(1, 3)).tcount(), 1)
        self.assertEqual(PauliExponential("X", new_var("theta", is_bool=False)).tcount(), 1)

    def test_remap_and_tensor(self):
        angle = Fraction(1, 4)
        rotation = PauliExponential("-XYI", angle)
        source = Circuit(3)
        source.add_gate(rotation)
        dest = Circuit(4)
        dest.add_circuit(source, mask=[3, 0, 1])
        self.assertEqual(dest.gates, [PauliExponential("-YIIX", angle)])
        np.testing.assert_allclose(dest.to_matrix(), exponential("YIIX", -angle), atol=1e-12)
        combined = Circuit(2) @ rotation
        self.assertEqual(combined.qubits, 5)
        np.testing.assert_allclose(combined.to_matrix(), exponential("IIXYI", -angle), atol=1e-12)

    def test_basic_gates_and_exports_up_to_global_phase(self):
        for pauli, axes, angle in (("-XY", "XY", Fraction(-1, 4)),
                                  ("-Z", "Z", 1), ("II", "II", Fraction(-1, 3))):
            with self.subTest(pauli=pauli, angle=angle):
                circuit = Circuit(len(axes))
                circuit.add_gate(PauliExponential(pauli, angle))
                alternatives = (
                    circuit.to_basic_gates(),
                    Circuit.from_qasm(circuit.to_qasm(version=2)),
                    Circuit.from_qasm(circuit.to_qasm(version=3)),
                    Circuit.from_qc(circuit.to_qc()),
                    Circuit.from_quipper(circuit.to_quipper()),
                )
                expected = circuit.to_matrix()
                for alternative in alternatives:
                    self.assertEqual(alternative.tcount(), circuit.tcount())
                    actual = alternative.to_matrix()
                    phase = np.vdot(expected, actual) / expected.shape[0]
                    self.assertAlmostEqual(abs(phase), 1)
                    np.testing.assert_allclose(actual, phase * expected, atol=1e-12)

    def test_gadget_phase_matches_phase_gadget(self):
        for sign, angle in (("+", 0), ("+", Fraction(9, 4)), ("-", Fraction(1, 4))):
            with self.subTest(sign=sign, angle=angle):
                rotation = PauliExponential(sign + "IZIZI", angle, gadget_phase=True)
                gadget = PhaseGadget(-angle if sign == "-" else angle, 1, 3)
                circuit = Circuit(5)
                circuit.add_gate(gadget)
                self.assertEqual(rotation.to_graph().to_dict(), circuit.to_graph().to_dict())
                self.assertEqual(rotation.to_basic_gates(), gadget.to_basic_gates())
                self.assertEqual(rotation.to_qasm(), gadget.to_qasm())
                self.assertEqual(rotation.to_qc(), gadget.to_qc())
                self.assertEqual(rotation.to_quipper(), gadget.to_quipper())

    def test_gadget_phase_mixed_and_identity_strings(self):
        angle = Fraction(1, 4)
        for axes in ("XY", "II"):
            with self.subTest(axes=axes):
                rotation = PauliExponential("-" + axes, angle, gadget_phase=True)
                expected = np.exp(-1j * np.pi / 8) * exponential(axes, -angle)
                np.testing.assert_allclose(rotation.to_graph().to_matrix(), expected, atol=1e-12)
                np.testing.assert_allclose(rotation.to_adjoint().to_graph().to_matrix(),
                                           expected.conj().T, atol=1e-12)

    def test_graph_uses_one_gadget(self):
        graph = PauliExponential("XIY", Fraction(1, 3)).to_graph()
        leaves = [v for v in graph.vertices() if graph.type(v) == VertexType.Z
                  and graph.vertex_degree(v) == 1 and graph.phase(v) == Fraction(1, 3)]
        self.assertEqual(len(leaves), 1)
        hub, = graph.neighbors(leaves[0])
        self.assertEqual(graph.vertex_degree(hub), 3)

    def test_symbolic_angle_and_global_phase(self):
        angle = new_var("theta", is_bool=False)
        value = Fraction(2, 3)
        for axes in ("XY", "II"):
            with self.subTest(axes=axes):
                graph = PauliExponential("-" + axes, angle).to_graph()
                bound = graph.substitute_variables({"theta": value})
                np.testing.assert_allclose(bound.to_matrix(), exponential(axes, -value), atol=1e-12)

    def test_rejects_malformed_pauli_strings(self):
        for text in ("", "-", "XX-", "A", "++X"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                PauliExponential(text, Fraction(1, 3))
