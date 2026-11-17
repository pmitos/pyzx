# Copyright (C) 2026 - Piotr Bartosz Mitosek
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Pauli exponentials and their ZX representation."""

from dataclasses import dataclass
from fractions import Fraction

from .circuit import Circuit, PhaseGadget
from .graph.base import BaseGraph
from .symbolic import Poly
from .utils import FractionLike, assert_phase_real, normalize_phase


@dataclass(frozen=True, slots=True)
class PauliExponential:
    r"""The unitary :math:`\exp(-i\pi a P/2)` for a signed Pauli tensor.

    ``pauli`` specifies :math:`P` as a nonempty string of ``I``, ``X``, ``Y``
    and ``Z``, with an optional leading sign. The leftmost axis is wire zero,
    and ``I`` leaves that wire in the register. ``angle`` is :math:`a`, in
    multiples of :math:`\pi`, and may be symbolic.
    """

    pauli: str
    angle: FractionLike

    def __post_init__(self) -> None:
        if not isinstance(self.pauli, str):
            raise TypeError("pauli must be a string")
        pauli = self.pauli if self.pauli.startswith(("+", "-")) else "+" + self.pauli
        if len(pauli) == 1 or any(axis not in "IXYZ" for axis in pauli[1:]):
            raise ValueError("pauli must contain I, X, Y or Z axes after an optional sign")
        angle = normalize_phase(self.angle)
        if not isinstance(angle, (int, Fraction, Poly)):
            raise TypeError("angle must be an integer, Fraction or symbolic phase")
        assert_phase_real(angle)
        object.__setattr__(self, "pauli", pauli)
        object.__setattr__(self, "angle", angle)

    def to_graph(self, backend: str | None = None) -> BaseGraph:
        """Return a ZX diagram using a phase gadget and local basis changes.

        The diagram includes the global phase of the exponential.
        """
        axes = self.pauli[1:]
        angle = -self.angle if self.pauli[0] == "-" else self.angle
        circuit = Circuit(len(axes))
        targets = [q for q, axis in enumerate(axes) if axis != "I"]
        if targets and angle != 0:
            for q in targets:
                if axes[q] == "Y":
                    circuit.add_gate("S", q, adjoint=True)
                if axes[q] in "XY":
                    circuit.add_gate("HAD", q)
            inverse_basis = circuit.adjoint()
            circuit.add_gate(PhaseGadget(angle, *targets))
            circuit.add_circuit(inverse_basis)

        graph = circuit.to_graph(backend=backend)
        # A phase gadget has eigenvalues 1 and exp(i*pi*angle).
        graph.scalar.add_phase(Fraction(-1, 2) * angle)
        return graph
