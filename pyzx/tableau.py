# Copyright (C) 2026 - Piotr Bartosz Mitosek
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Signed Pauli strings and tableau representations of Clifford isometries.

The representation preserves logical Pauli images and output stabilizers up
to global phase. Circuit conversion is a reference implementation: isometries
are completed to unitary tableaux and synthesized by symplectic elimination.
It does not optimize gate count, connectivity, depth or the choice of completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .circuit import Circuit
from .circuit.gates import CNOT, CZ, HAD, SWAP, XCX, Y, InitAncilla, XPhase, ZPhase

__all__ = ["PauliString", "CliffordMap", "CliffordMapError"]

_Row = tuple[int, int, int]


class CliffordMapError(ValueError):
    """An invalid Clifford tableau or an unsupported circuit conversion."""


@dataclass(frozen=True, slots=True)
class PauliString:
    """Hermitian signed Pauli tensor, packed X/Z supports; bit q is wire q."""

    n_qubits: int
    x: int
    z: int
    sign: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_qubits, bool)
            or not isinstance(self.n_qubits, int)
            or self.n_qubits < 0
        ):
            raise ValueError("n_qubits must be a nonnegative integer")
        for value in (self.x, self.z):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < (1 << self.n_qubits)
            ):
                raise ValueError("Pauli support exceeds the register")
        if (
            isinstance(self.sign, bool)
            or not isinstance(self.sign, int)
            or self.sign not in (-1, 1)
        ):
            raise ValueError("a Hermitian Pauli sign must be +1 or -1")

    def commutes(self, other: PauliString) -> bool:
        if self.n_qubits != other.n_qubits:
            raise ValueError("Paulis act on different registers")
        return not (((self.x & other.z).bit_count() + (self.z & other.x).bit_count()) & 1)

    def __str__(self) -> str:
        return ("-" if self.sign < 0 else "+") + "".join(
            "IXZY"[((self.x >> q) & 1) + 2 * ((self.z >> q) & 1)] for q in range(self.n_qubits)
        )

    def to_matrix(self, *, max_qubits: int = 8) -> np.ndarray:
        _dense_limit(self.n_qubits, max_qubits)
        matrices = (
            np.eye(2),
            np.array([[0, 1], [1, 0]]),
            np.diag([1, -1]),
            np.array([[0, -1j], [1j, 0]]),
        )
        result = np.array([[complex(self.sign)]])
        for q in range(self.n_qubits):
            result = np.kron(result, matrices[((self.x >> q) & 1) + 2 * ((self.z >> q) & 1)])
        return result


def _dense_limit(n: int, limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0 or n > limit:
        raise ValueError(f"dense verification of {n} qubits exceeds max_qubits={limit}")


@dataclass(frozen=True)
class CliffordMap:
    """Clifford isometry specified up to global phase, without gate synthesis.

    Logical images obey ``x_images[q] C = C X_q`` and likewise for Z.
    Independent signed ``stabilizers`` obey ``S C = C`` and specify the
    output subspace. There is one stabilizer per extra output. With no
    stabilizers this is the usual unitary Clifford tableau.
    """

    x_images: tuple[PauliString, ...]
    z_images: tuple[PauliString, ...]
    stabilizers: tuple[PauliString, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_images", tuple(self.x_images))
        object.__setattr__(self, "z_images", tuple(self.z_images))
        object.__setattr__(self, "stabilizers", tuple(self.stabilizers))
        n, m = self.n_inputs, self.n_outputs
        if len(self.z_images) != n or any(
            p.n_qubits != m for p in self.x_images + self.z_images + self.stabilizers
        ):
            raise CliffordMapError("Clifford generator images have incompatible dimensions")
        for i in range(n):
            for j in range(n):
                if (
                    not self.x_images[i].commutes(self.x_images[j])
                    or not self.z_images[i].commutes(self.z_images[j])
                    or self.x_images[i].commutes(self.z_images[j]) != (i != j)
                ):
                    raise CliffordMapError(
                        "Clifford images violate the Pauli commutation relations"
                    )
        basis: dict[int, int] = {}
        for s in self.stabilizers:
            if any(not s.commutes(p) for p in self.x_images + self.z_images + self.stabilizers):
                raise CliffordMapError("Clifford stabilizers violate the Pauli commutation relations")
            support = s.x | (s.z << m)
            while support:
                pivot = support & -support
                if pivot not in basis:
                    basis[pivot] = support
                    break
                support ^= basis[pivot]
            if not support:
                raise CliffordMapError("Clifford stabilizers must be independent")

    @property
    def n_inputs(self) -> int:
        return len(self.x_images)

    @property
    def n_outputs(self) -> int:
        return self.n_inputs + len(self.stabilizers)

    @property
    def n_qubits(self) -> int:
        """Size of the output register."""
        return self.n_outputs

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> CliffordMap:
        """Propagate signed Pauli generators through a basic Clifford circuit.

        Supported gates are HAD, CNOT/CX, CZ, XCX, SWAP, Y, and X/Z phases
        that are exact multiples of pi/2 (including X, Z, S, Sdg and SX).
        Leading InitAncilla gates may prepare +, -, 0 or 1 on consecutive
        wires after the open input wires. Other gates and circuit-level
        initialization/postselection flags are rejected, not approximated.
        """
        n = circuit.qubits
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise CliffordMapError("the input count must be a nonnegative integer")
        if circuit.bits or circuit._postselect_qubits is not None:
            raise CliffordMapError("classical wires and postselection are unsupported")
        if circuit._initialize_qubits is not None and any(circuit._initialize_qubits):
            raise CliffordMapError("use leading InitAncilla gates for prepared input wires")
        preparations: list[InitAncilla] = []
        for gate in circuit.gates:
            if not isinstance(gate, InitAncilla):
                break
            if (isinstance(gate.label, bool) or not isinstance(gate.label, int)
                    or gate.label != n + len(preparations)):
                raise CliffordMapError("ancillas must have consecutive labels after the inputs")
            preparations.append(gate)
        m = n + len(preparations)
        rows: list[_Row] = [(1 << q, 0, 0) for q in range(n)]
        rows.extend((0, 1 << q, 0) for q in range(n))
        for gate in preparations:
            bit = 1 << gate.label
            if gate.state not in ("+", "-", "0", "1"):
                raise CliffordMapError("unsupported ancilla state")
            rows.append((bit, 0, int(gate.state == "-")) if gate.state in ("+", "-")
                        else (0, bit, int(gate.state == "1")))

        def wire(q: int) -> int:
            if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < m:
                raise CliffordMapError("gate target is outside the output register")
            return q

        for gate in circuit.gates[len(preparations):]:
            if isinstance(gate, InitAncilla):
                raise CliffordMapError("ancilla preparations must precede every other gate")
            if isinstance(gate, (CNOT, CZ)):
                a, b = wire(gate.control), wire(gate.target)
                if a == b:
                    raise CliffordMapError("a two-qubit gate needs distinct targets")
                if isinstance(gate, SWAP):
                    _conjugate(rows, "SWAP", a, b)
                elif isinstance(gate, XCX):
                    _conjugate(rows, "HAD", a)
                    _conjugate(rows, "CNOT", a, b)
                    _conjugate(rows, "HAD", a)
                elif isinstance(gate, CNOT):
                    _conjugate(rows, "CNOT", a, b)
                else:
                    _conjugate(rows, "HAD", b)
                    _conjugate(rows, "CNOT", a, b)
                    _conjugate(rows, "HAD", b)
            elif isinstance(gate, HAD):
                _conjugate(rows, "HAD", wire(gate.target))
            elif isinstance(gate, (XPhase, ZPhase)):
                if not isinstance(gate.phase, (int, float, Fraction)):
                    raise CliffordMapError("Clifford phases must be numeric multiples of pi/2")
                try:
                    turns = 2 * (Fraction(gate.phase) % 2)
                except (ValueError, OverflowError) as error:
                    raise CliffordMapError("Clifford phases must be finite") from error
                if turns.denominator != 1:
                    raise CliffordMapError("non-Clifford phase; expected a multiple of pi/2")
                q = wire(gate.target)
                if isinstance(gate, XPhase):
                    _conjugate(rows, "HAD", q)
                for _ in range(int(turns)):
                    _conjugate(rows, "S", q)
                if isinstance(gate, XPhase):
                    _conjugate(rows, "HAD", q)
            elif isinstance(gate, Y):
                q = wire(gate.target)
                _conjugate(rows, "NOT", q)
                _conjugate(rows, "Z", q)
            else:
                raise CliffordMapError(f"unsupported Clifford conversion gate: {gate.name}")
        paulis = tuple(PauliString(m, x, z, -1 if negative else 1)
                       for x, z, negative in rows)
        return cls(paulis[:n], paulis[n:2 * n], paulis[2 * n:])

    def to_circuit(self) -> Circuit:
        """Reference synthesis, preparing extra output wires in ket plus.

        Complete the isometry to a unitary tableau, then perform signed
        symplectic elimination. This chooses an arbitrary completion and
        does not optimize gate count, depth or hardware connectivity.

        ``Circuit.qubits`` counts the input wires; ``InitAncilla`` gates add
        wires numbered from ``n_inputs`` to ``n_outputs - 1``.
        """
        unitary = _synthesise_clifford(_complete_clifford(self))
        if not self.stabilizers:
            return unitary
        circuit = Circuit(self.n_inputs)
        circuit.gates.extend(InitAncilla(q) for q in range(self.n_inputs, self.n_outputs))
        circuit.gates.extend(unitary.gates)
        return circuit

    def to_matrix(self, *, max_qubits: int = 8) -> np.ndarray:
        """Small-case verification only; choose an arbitrary global phase."""
        n = self.n_qubits
        _dense_limit(n, max_qubits)
        dim = 1 << n
        zs = [p.to_matrix(max_qubits=max_qubits) for p in self.z_images + self.stabilizers]
        xs = [p.to_matrix(max_qubits=max_qubits) for p in self.x_images]
        for seed in range(dim):
            zero = np.eye(dim, dtype=complex)[:, seed]
            for z in zs:
                zero = (zero + z @ zero) / 2
            norm = np.linalg.norm(zero)
            if norm > 1e-10:  # cap 8: a nonzero stabilizer amplitude is >= 1/16
                zero /= norm
                break
        else:
            raise CliffordMapError("inconsistent Clifford stabilizers")
        columns = []
        for basis in range(1 << self.n_inputs):
            column = zero.copy()
            for q, x in enumerate(xs):
                if basis & (1 << (self.n_inputs - 1 - q)):
                    column = x @ column
            columns.append(column)
        return np.column_stack(columns)


def _multiply(a: _Row, b: _Row) -> _Row:
    """Multiply commuting Hermitian packed rows (x,z,negative), exactly."""
    x, z = a[0] ^ b[0], a[1] ^ b[1]
    phase = (
        (a[0] & a[1]).bit_count()
        + (b[0] & b[1]).bit_count()
        - (x & z).bit_count()
        + 2 * (a[1] & b[0]).bit_count()
    )
    if phase & 1:
        raise CliffordMapError("anticommuting stabilizer row product")
    return x, z, a[2] ^ b[2] ^ ((phase // 2) & 1)


def _conjugate(
    rows: list[_Row], name: str, a: int, b: int | None = None, *, adjoint: bool = False,
) -> None:
    """Conjugate packed Hermitian rows by one elementary Clifford gate."""
    bit = 1 << a
    other = 0 if b is None else 1 << b
    for j, (x, z, negative) in enumerate(rows):
        xa, za = bool(x & bit), bool(z & bit)
        if name == "HAD":
            negative ^= xa and za
            if xa != za:
                x ^= bit
                z ^= bit
        elif name == "S":
            negative ^= xa and (not za if adjoint else za)
            if xa:
                z ^= bit
        elif name == "Z":
            negative ^= xa
        elif name == "NOT":
            negative ^= za
        elif name == "SWAP":
            if xa != bool(x & other):
                x ^= bit | other
            if za != bool(z & other):
                z ^= bit | other
        elif name == "CNOT":
            xb, zb = bool(x & other), bool(z & other)
            negative ^= xa and zb and (xb == za)
            if xa:
                x ^= other
            if zb:
                z ^= bit
        else:
            raise CliffordMapError(f"unsupported elementary Clifford gate: {name}")
        rows[j] = x, z, int(negative)


def _complete_clifford(clifford: CliffordMap) -> CliffordMap:
    """Complete an isometry to a unitary tableau, only for optional synthesis.

    Extra X images are the output stabilizers, corresponding to ket-plus
    preparations. Solve all missing Z partners together, then remove their
    mutual anticommutations by adding earlier stabilizers. The completion
    outside the image subspace is arbitrary.
    """
    if not clifford.stabilizers:
        return clifford
    n, m = clifford.n_inputs, clifford.n_outputs
    xs = clifford.x_images + clifford.stabilizers
    rows = [p.z | (p.x << m) | (1 << (2 * m + j - n) if j >= n else 0)
            for j, p in enumerate(xs)]
    rows.extend(p.z | (p.x << m) for p in clifford.z_images)
    pivots = []
    rank = 0
    for col in range(2 * m):
        pivot = next((j for j in range(rank, len(rows)) if rows[j] & (1 << col)), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for j in range(len(rows)):
            if j != rank and rows[j] & (1 << col):
                rows[j] ^= rows[rank]
        pivots.append(col)
        rank += 1
    mask = (1 << m) - 1
    partners: list[PauliString] = []
    for i in range(m - n):
        support = sum(1 << col for col, row in zip(pivots, rows) if row & (1 << (2 * m + i)))
        partner = PauliString(m, support & mask, support >> m)
        for j, previous in enumerate(partners):
            if not partner.commutes(previous):
                s = clifford.stabilizers[j]
                x, z, sign = _multiply((partner.x, partner.z, int(partner.sign < 0)),
                                       (s.x, s.z, int(s.sign < 0)))
                partner = PauliString(m, x, z, -1 if sign else 1)
        partners.append(partner)
    return CliffordMap(xs, clifford.z_images + tuple(partners))


def _synthesise_clifford(clifford: CliffordMap) -> Circuit:
    """Reduce signed X/Z images to identity by output Clifford operations.

    For each wire fix its X image, then its Z image. Symplectic commutation
    ensures later images cannot act on the already fixed wires. Reversing
    and adjoining the resulting gates gives the original Clifford.
    """
    n = clifford.n_qubits
    rows = [(p.x, p.z, int(p.sign < 0)) for p in clifford.x_images + clifford.z_images]
    reduction = Circuit(n)

    def apply(name: str, a: int, b: int | None = None, *, adjoint: bool = False) -> None:
        if b is None:
            if name == "S":
                reduction.add_gate(name, a, adjoint=adjoint)
            else:
                reduction.add_gate(name, a)
        else:
            reduction.add_gate(name, a, b)
        _conjugate(rows, name, a, b, adjoint=adjoint)

    for q in range(n):
        bit = 1 << q
        pivot = next(j for j in range(q, n) if (rows[q][0] | rows[q][1]) & (1 << j))
        if pivot != q:
            apply("SWAP", q, pivot)
        for j in range(q, n):
            mask = 1 << j
            if rows[q][1] & mask:
                if rows[q][0] & mask:
                    apply("S", j, adjoint=True)
                else:
                    apply("HAD", j)
        for j in range(q + 1, n):
            if rows[q][0] & (1 << j):
                apply("CNOT", q, j)
        if rows[q][2]:
            apply("Z", q)
        if rows[n + q][0] & bit:
            apply("HAD", q)
            apply("S", q)
            apply("HAD", q)
        for j in range(q + 1, n):
            mask = 1 << j
            if rows[n + q][0] & mask:
                if rows[n + q][1] & mask:
                    apply("S", j, adjoint=True)
                apply("HAD", j)
        for j in range(q + 1, n):
            if rows[n + q][1] & (1 << j):
                apply("CNOT", j, q)
        if rows[n + q][2]:
            apply("NOT", q)
    return reduction.adjoint()
