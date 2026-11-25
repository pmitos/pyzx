"""Tableau semantics checked independently against circuit and Pauli matrices."""

import itertools
import random
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np

import pyzx as zx
from pyzx.circuit.gates import InitAncilla, Measurement, Reset
from pyzx.symbolic import new_var
from pyzx.tableau import CliffordMap, CliffordMapError, PauliString


class TestPauliString(unittest.TestCase):
    def test_signed_tensor_convention(self):
        pauli = PauliString(3, 1, 4, -1)
        self.assertEqual(str(pauli), "-XIZ")
        x = np.array([[0, 1], [1, 0]])
        z = np.diag([1, -1])
        np.testing.assert_array_equal(pauli.to_matrix(), -np.kron(np.kron(x, np.eye(2)), z))
        np.testing.assert_array_equal(PauliString(1, 1, 1).to_matrix(),
                                      np.array([[0, -1j], [1j, 0]]))

    def test_commutation_matches_matrices(self):
        strings = [PauliString(2, x, z) for x, z in itertools.product(range(4), repeat=2)]
        for a, b in itertools.product(strings, repeat=2):
            product = a.to_matrix() @ b.to_matrix()
            reverse = b.to_matrix() @ a.to_matrix()
            np.testing.assert_array_equal(product, reverse if a.commutes(b) else -reverse)
        with self.assertRaises(ValueError):
            strings[0].commutes(PauliString(1, 0, 0))

    def test_invalid_packed_rows_and_dense_limit(self):
        for args in [(-1, 0, 0), (True, 0, 0), (1, -1, 0), (1, 2, 0),
                     (1, 0, 2), (1, 0.5, 0), (1, 0, 0, 0), (1, 0, 0, True)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                PauliString(*args)
        for limit in [-1, True, 0.5, 1]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                PauliString(2, 0, 0).to_matrix(max_qubits=limit)


class TestCliffordMap(unittest.TestCase):
    def assert_same(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        scale = np.vdot(expected, actual) / np.vdot(expected, expected)
        self.assertGreater(abs(scale), 1e-10)
        np.testing.assert_allclose(actual, scale * expected, atol=1e-9, rtol=0)

    def check_relations(self, tableau):
        matrix = tableau.to_matrix()
        np.testing.assert_allclose(matrix.conj().T @ matrix,
                                   np.eye(2 ** tableau.n_inputs), atol=1e-9, rtol=0)
        for q, (x, z) in enumerate(zip(tableau.x_images, tableau.z_images)):
            np.testing.assert_allclose(x.to_matrix() @ matrix,
                                       matrix @ PauliString(tableau.n_inputs, 1 << q, 0).to_matrix(),
                                       atol=1e-9, rtol=0)
            np.testing.assert_allclose(z.to_matrix() @ matrix,
                                       matrix @ PauliString(tableau.n_inputs, 0, 1 << q).to_matrix(),
                                       atol=1e-9, rtol=0)
        for stabilizer in tableau.stabilizers:
            np.testing.assert_allclose(stabilizer.to_matrix() @ matrix, matrix, atol=1e-9, rtol=0)

    def test_public_exports_and_empty_map(self):
        self.assertIs(zx.CliffordMap, CliffordMap)
        self.assertIs(zx.PauliString, PauliString)
        self.assertIs(zx.CliffordMapError, CliffordMapError)
        empty = CliffordMap.from_circuit(zx.Circuit(0))
        self.assertEqual((empty.n_inputs, empty.n_outputs), (0, 0))
        np.testing.assert_array_equal(empty.to_matrix(), [[1]])
        self.assert_same(empty.to_matrix(), empty.to_circuit().to_matrix())

    def test_signed_single_qubit_images(self):
        for name, kwargs, x, z in [
            ("HAD", {}, "+Z", "+X"), ("S", {}, "+Y", "+Z"),
            ("S", {"adjoint": True}, "-Y", "+Z"),
            ("NOT", {}, "+X", "-Z"), ("Z", {}, "-X", "+Z"),
            ("Y", {}, "-X", "-Z"),
        ]:
            with self.subTest(name=name, kwargs=kwargs):
                circuit = zx.Circuit(1)
                circuit.add_gate(name, 0, **kwargs)
                tableau = CliffordMap.from_circuit(circuit)
                self.assertEqual(tuple(map(str, tableau.x_images)), (x,))
                self.assertEqual(tuple(map(str, tableau.z_images)), (z,))
                self.assert_same(circuit.to_matrix(), tableau.to_matrix())

    def test_exact_half_pi_phases(self):
        for name, phase in itertools.product(("XPhase", "ZPhase"),
                                              [Fraction(k, 2) for k in range(-5, 7)]):
            with self.subTest(name=name, phase=phase):
                circuit = zx.Circuit(1)
                circuit.add_gate(name, 0, phase=phase)
                tableau = CliffordMap.from_circuit(circuit)
                self.assert_same(circuit.to_matrix(), tableau.to_matrix())
                self.assert_same(circuit.to_matrix(), tableau.to_circuit().to_matrix())

    def test_two_qubit_gate_conjugation(self):
        for name, targets in itertools.product(("CNOT", "CX", "CZ", "XCX", "SWAP"),
                                                [(0, 1), (1, 0)]):
            with self.subTest(name=name, targets=targets):
                circuit = zx.Circuit(2)
                circuit.add_gate(name, *targets)
                tableau = CliffordMap.from_circuit(circuit)
                self.assert_same(circuit.to_matrix(), tableau.to_matrix())
                self.assert_same(circuit.to_matrix(), tableau.to_circuit().to_matrix())

    def test_encoder_and_signed_bell_state(self):
        encoder = zx.Circuit(1)
        encoder.add_gate(InitAncilla(1, state="0"))
        encoder.add_gate("CNOT", 0, 1)
        tableau = CliffordMap.from_circuit(encoder)
        self.assertEqual(tuple(map(str, tableau.x_images)), ("+XX",))
        self.assertEqual(tuple(map(str, tableau.z_images)), ("+ZI",))
        self.assertEqual(tuple(map(str, tableau.stabilizers)), ("+ZZ",))
        self.assert_same(np.array([[1, 0], [0, 0], [0, 0], [0, 1]]), tableau.to_matrix())
        signed_bell = CliffordMap((), (), (PauliString(2, 3, 0, -1), PauliString(2, 0, 3)))
        self.assert_same(np.array([[1], [0], [0], [-1]]), signed_bell.to_matrix())
        for example in [tableau, signed_bell]:
            self.check_relations(example)
            self.assert_same(example.to_matrix(), example.to_circuit().to_matrix())

    def test_all_preparation_states(self):
        for state, stabilizer in [("+", "+X"), ("-", "-X"), ("0", "+Z"), ("1", "-Z")]:
            with self.subTest(state=state):
                circuit = zx.Circuit(0)
                circuit.add_gate(InitAncilla(0, state=state))
                tableau = CliffordMap.from_circuit(circuit)
                self.assertEqual(tuple(map(str, tableau.stabilizers)), (stabilizer,))
                self.assert_same(circuit.to_matrix(), tableau.to_matrix())
                self.assert_same(circuit.to_matrix(), tableau.to_circuit().to_matrix())

    def test_random_clifford_unitaries_and_isometries(self):
        rng = random.Random(20261125)
        for case in range(32):
            inputs, extra = case % 4, case % 3
            outputs = inputs + extra
            circuit = zx.Circuit(inputs)
            for q in range(inputs, outputs):
                circuit.add_gate(InitAncilla(q, rng.choice(("+", "-", "0", "1"))))
            for _ in range(18 if outputs else 0):
                name = rng.choice(("HAD", "S", "NOT", "Z", "XPhase", "ZPhase")
                                  + (("CNOT", "CZ", "SWAP", "XCX") if outputs > 1 else ()))
                if name in ("CNOT", "CZ", "SWAP", "XCX"):
                    circuit.add_gate(name, *rng.sample(range(outputs), 2))
                elif name in ("XPhase", "ZPhase"):
                    circuit.add_gate(name, rng.randrange(outputs), phase=Fraction(rng.randrange(-4, 5), 2))
                else:
                    circuit.add_gate(name, rng.randrange(outputs))
            with self.subTest(case=case):
                tableau = CliffordMap.from_circuit(circuit)
                self.assertEqual((tableau.n_inputs, tableau.n_outputs), (inputs, outputs))
                self.assert_same(circuit.to_matrix(), tableau.to_matrix())
                synthesized = tableau.to_circuit()
                self.assertEqual(synthesized.qubits, inputs)
                self.assertEqual([gate.state for gate in synthesized.gates
                                  if isinstance(gate, InitAncilla)], ["+"] * extra)
                self.assert_same(circuit.to_matrix(), synthesized.to_matrix())
                self.check_relations(tableau)

    def test_packed_supports_beyond_64_bits_and_no_dense_synthesis(self):
        circuit = zx.Circuit(70)
        circuit.add_gate("CNOT", 0, 69)
        circuit.add_gate("HAD", 69)
        circuit.add_gate("S", 65, adjoint=True)
        with patch.object(CliffordMap, "to_matrix", side_effect=AssertionError("dense evaluation")):
            tableau = CliffordMap.from_circuit(circuit)
            synthesized = tableau.to_circuit()
            self.assertEqual(CliffordMap.from_circuit(synthesized), tableau)
        self.assertEqual(tableau.x_images[0], PauliString(70, 1, 1 << 69))
        with self.assertRaisesRegex(ValueError, "dense verification"):
            tableau.to_matrix()

    def test_invalid_tableau_relations(self):
        for args in [
            ((PauliString(1, 1, 0),), (), ()),
            ((PauliString(1, 1, 0),), (PauliString(1, 1, 0),), ()),
            ((), (), (PauliString(1, 0, 0),)),
            ((), (), (PauliString(2, 1, 0), PauliString(2, 1, 0, -1))),
            ((), (), (PauliString(2, 1, 0), PauliString(2, 0, 1))),
            ((PauliString(2, 1, 0),), (PauliString(2, 0, 1),), (PauliString(2, 1, 0),)),
        ]:
            with self.subTest(args=args), self.assertRaises(CliffordMapError):
                CliffordMap(*args)

    def test_unsupported_operations_and_no_phase_rounding(self):
        cases = [
            ("T", (0,), {}), ("ZPhase", (0,), {"phase": Fraction(1, 3)}),
            ("XPhase", (0,), {"phase": 0.5000000000000001}),
            ("ZPhase", (0,), {"phase": new_var("tableau_phase", is_bool=False)}),
            ("ZPhase", (0,), {"phase": float("nan")}),
            ("ZPhase", (0,), {"phase": float("inf")}),
        ]
        for name, targets, kwargs in cases:
            circuit = zx.Circuit(1)
            circuit.add_gate(name, *targets, **kwargs)
            with self.subTest(name=name, kwargs=kwargs), self.assertRaises(CliffordMapError):
                CliffordMap.from_circuit(circuit)
        for gate in [Reset(0), Measurement(0)]:
            circuit = zx.Circuit(1)
            circuit.add_gate(gate)
            with self.subTest(gate=gate), self.assertRaises(CliffordMapError):
                CliffordMap.from_circuit(circuit)

    def test_invalid_registers_and_preparation_order(self):
        circuits = [zx.Circuit(-1), zx.Circuit(True), zx.Circuit(1, bit_amount=1)]
        initialized = zx.Circuit(1)
        initialized.initialize_qubits([True])
        circuits.append(initialized)
        postselected = zx.Circuit(1)
        postselected.postselect_qubits([])
        circuits.append(postselected)
        for prep in [InitAncilla(2), InitAncilla(0), InitAncilla(True), InitAncilla(1.0)]:
            circuit = zx.Circuit(1)
            circuit.add_gate(prep)
            circuits.append(circuit)
        late = zx.Circuit(1)
        late.add_gate("HAD", 0)
        late.add_gate(InitAncilla(1))
        circuits.append(late)
        for name, targets in [("HAD", (-1,)), ("HAD", (1,)),
                              ("CNOT", (0, 0)), ("CNOT", (0, 2))]:
            circuit = zx.Circuit(1)
            circuit.add_gate(name, *targets)
            circuits.append(circuit)
        for circuit in circuits:
            with self.subTest(circuit=circuit), self.assertRaises(CliffordMapError):
                CliffordMap.from_circuit(circuit)


if __name__ == "__main__":
    unittest.main()
