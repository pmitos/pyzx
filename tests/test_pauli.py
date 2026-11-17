import unittest
from fractions import Fraction

import numpy as np

from pyzx import PauliExponential, VertexType
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
