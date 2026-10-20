# Copyright (C) 2026 - Piotr Bartosz Mitosek
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Pauli-flow circuit extraction following Simmons, arXiv:2109.05654.

Lemma C.2 gives output Pauli strings for the planar rotations. Theorem C.12
separates those rotations from a residual Clifford map. Here that map is
reconstructed by projecting its Choi stabilizers, with packed row supports
and incidence columns. A supplied focused flow is validated and reused.
Both extraction entry points return a PauliCircuit; elementary-gate synthesis
is an explicit, optional conversion through PauliCircuit.to_circuit().

Extraction does not construct dense matrices. The current domain has equal
input/output counts, rational angles and no discarded qubits. Results agree
up to a nonzero global scalar. The existing frontier extractor is separate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np

from .circuit import Circuit
from .circuit.gates import ParityPhase
from .flow import gflow, pauli_flow
from .graph.base import BaseGraph
from .symbolic import Poly
from .utils import EdgeType, VertexType

__all__ = ["Measurement", "PauliPattern", "PauliString", "CliffordMap",
           "PauliExponential", "PauliCircuit", "PauliExtractionError",
           "extract_pauli", "extract_circuit_pauli"]

_LABELS = frozenset({"XY", "XZ", "YZ", "X", "Y", "Z"})
_PLANES = frozenset({"XY", "XZ", "YZ"})
Flow = tuple[dict[Any, int], dict[Any, set[Any]]]
_Row = tuple[int, int, int]


class PauliExtractionError(ValueError):
    """Invalid pattern, invalid witness, absent flow or unsupported domain."""


def _bits(value: int) -> Iterator[int]:
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def _phase(value: Fraction | int | float) -> Fraction:
    if isinstance(value, Poly):
        raise PauliExtractionError("Pauli extraction requires rational, non-symbolic phases")
    return Fraction(value) % 2


@dataclass(frozen=True, slots=True)
class Measurement:
    """Positive-outcome measurement in Simmons's conventions, in units of pi.

    XY uses ``(|0> + exp(i*pi*angle)|1>)/sqrt(2)``; XZ uses
    ``cos(pi*angle/2)|0> + sin(pi*angle/2)|1>``; YZ replaces sin by i*sin.
    Axis labels use angle 0 or 1 for the positive or negative eigenstate.
    """

    label: str
    angle: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise PauliExtractionError(f"unknown measurement label {self.label!r}")
        object.__setattr__(self, "angle", _phase(self.angle))
        if self.label not in _PLANES and self.angle.denominator != 1:
            raise PauliExtractionError("an axis measurement angle must be 0 or 1")

    def infer_pauli(self) -> Measurement:
        """Replace a planar measurement at an exact Clifford angle by its axis."""
        if self.label not in _PLANES or self.angle.denominator > 2:
            return self
        labels = {"XY": ("X", "Y"), "XZ": ("Z", "X"), "YZ": ("Z", "Y")}
        quarter = int(2 * self.angle)
        return Measurement(labels[self.label][quarter % 2], Fraction(quarter // 2))


@dataclass(frozen=True)
class PauliPattern:
    """A simple labelled open graph with ordered inputs and outputs.

    ``measurements`` covers exactly the non-outputs and supports all six labels.
    Boundary Hadamards are specified by wire index; output phases are applied
    before output Hadamards. Angles are rational multiples of pi.
    """

    vertices: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    measurements: Mapping[Any, Measurement]
    input_hadamards: tuple[int, ...] = ()
    output_hadamards: tuple[int, ...] = ()
    output_phases: tuple[Fraction, ...] = ()
    _direct_wires: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        vertices = tuple(self.vertices)
        positions = {v: i for i, v in enumerate(vertices)}
        if len(positions) != len(vertices):
            raise PauliExtractionError("duplicate vertex ids")
        for name in ("inputs", "outputs"):
            boundary = tuple(getattr(self, name))
            if len(set(boundary)) != len(boundary) or not set(boundary) <= positions.keys():
                raise PauliExtractionError(f"{name} must be distinct vertices")
            object.__setattr__(self, name, boundary)
        seen = set()
        edges = []
        for u, v in self.edges:
            if u not in positions or v not in positions or u == v:
                raise PauliExtractionError("the open graph must be simple")
            key = tuple(sorted((positions[u], positions[v])))
            if key in seen:
                raise PauliExtractionError("the open graph must be simple")
            seen.add(key)
            edges.append((u, v))
        measurements = dict(self.measurements)
        if set(measurements) != set(vertices) - set(self.outputs):
            raise PauliExtractionError("measurements must cover exactly the non-outputs")
        if not all(isinstance(m, Measurement) for m in measurements.values()):
            raise TypeError("measurements must contain Measurement values")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", tuple(edges))
        object.__setattr__(self, "measurements", MappingProxyType(measurements))
        for name, count in (("input_hadamards", len(self.inputs)),
                            ("output_hadamards", len(self.outputs))):
            wires = tuple(getattr(self, name))
            if len(set(wires)) != len(wires) or any(
                isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < count for q in wires
            ):
                raise PauliExtractionError(f"invalid {name} wire indices")
            object.__setattr__(self, name, wires)
        phases = tuple(_phase(a) for a in self.output_phases)
        if not phases:
            phases = (Fraction(0),) * len(self.outputs)
        if len(phases) != len(self.outputs):
            raise PauliExtractionError("one output phase is required per output")
        object.__setattr__(self, "output_phases", phases)

    @classmethod
    def from_graph(cls, graph: BaseGraph, *, infer_pauli: bool = True) -> PauliPattern:
        """Read a graph-like ZX diagram without mutating or simplifying it.

        A measured Z-spider of phase a represents the XY measurement at -a.
        Clifford angles can be inferred as signed X/Y measurements. Direct
        input-output wires and shared input/output spiders are supported.
        """
        if graph.is_hybrid() or graph.scalar.is_zero:
            raise PauliExtractionError("cannot extract a graph with grounds or a zero scalar")
        ins, outs = tuple(graph.inputs()), tuple(graph.outputs())
        boundaries = set(ins) | set(outs)
        if len(set(ins)) != len(ins) or len(set(outs)) != len(outs) or set(ins) & set(outs):
            raise PauliExtractionError("input/output boundary vertices must be distinct")
        vertices = tuple(graph.vertices())
        if boundaries != {v for v in vertices if graph.type(v) == VertexType.BOUNDARY}:
            raise PauliExtractionError("every boundary must be designated as an input or output")
        if any(graph.type(v) not in (VertexType.Z, VertexType.BOUNDARY) for v in vertices):
            raise PauliExtractionError("expected a graph-like diagram; run to_graph_like first")
        neighbors = {v: tuple(graph.neighbors(v)) for v in vertices}
        if any(graph.vertex_degree(b) != 1 or graph.phase(b) != 0 for b in boundaries):
            raise PauliExtractionError("boundary vertices must have degree one and phase zero")
        edges, seen = [], set()
        positions = {v: i for i, v in enumerate(vertices)}
        for e in graph.edges():
            u, v = graph.edge_st(e)
            key = tuple(sorted((positions[u], positions[v])))
            if u == v or key in seen:
                raise PauliExtractionError("expected a simple graph without parallel edges")
            seen.add(key)
            if u not in boundaries and v not in boundaries:
                if graph.edge_type(e) != EdgeType.HADAMARD:
                    raise PauliExtractionError("spiders must be connected by Hadamard edges")
                edges.append((u, v))
            elif graph.edge_type(e) not in (EdgeType.SIMPLE, EdgeType.HADAMARD):
                raise PauliExtractionError("unsupported boundary edge type")
        direct = []
        aliases = {}
        for b in ins:
            neighbor = neighbors[b][0]
            if neighbor in boundaries:
                if neighbor not in outs:
                    raise PauliExtractionError("a direct wire must connect an input to an output")
                aliases[neighbor] = b
                direct.append(b)
        for b in outs:
            if neighbors[b][0] in boundaries and b not in aliases:
                raise PauliExtractionError("a direct wire must connect an input to an output")
        def vertex(b: Any) -> Any:
            return aliases[b] if b in aliases else b if b in direct else neighbors[b][0]
        pattern_vertices = tuple(v for v in vertices if v not in boundaries) + tuple(direct)
        pattern_inputs = tuple(vertex(b) for b in ins)
        pattern_outputs = tuple(vertex(b) for b in outs)
        output_set = set(pattern_outputs)
        measurements = {}
        for v in pattern_vertices:
            if v not in output_set:
                phase = graph.phase(v)
                if isinstance(phase, Poly):
                    raise PauliExtractionError("Pauli extraction requires rational, non-symbolic phases")
                measurement = Measurement("XY", _phase(-phase))
                measurements[v] = measurement.infer_pauli() if infer_pauli else measurement
        ih = tuple(q for q, b in enumerate(ins) if b not in direct and
                   graph.edge_type(graph.edge(b, neighbors[b][0])) == EdgeType.HADAMARD)
        oh = tuple(q for q, b in enumerate(outs) if
                   graph.edge_type(graph.edge(b, neighbors[b][0])) == EdgeType.HADAMARD)
        phases = []
        for v in pattern_outputs:
            phase = 0 if v in direct else graph.phase(v)
            if isinstance(phase, Poly):
                raise PauliExtractionError("Pauli extraction requires rational, non-symbolic phases")
            phases.append(_phase(phase))
        return cls(pattern_vertices, tuple(edges), pattern_inputs, pattern_outputs,
                   measurements, ih, oh, tuple(phases), tuple(direct))

    def find_flow(self) -> Flow | None:
        """Find focused flow for the square domain using the public M/N equations.

        Equal input/output counts make M square, so its right inverse is unique.
        This is direct GF(2) inversion and a DAG check (Mitosek--Backens,
        DOI 10.1088/1751-8121/ae2999), without kernel adjustment or new dependencies.
        """
        if len(self.inputs) != len(self.outputs):
            raise PauliExtractionError("Pauli extraction requires equal input/output counts")
        pos, adjacency = _adjacency(self)
        measured = list(self.measurements)
        correctors = [v for v in self.vertices if v not in set(self.inputs)]
        indices = {v: j for j, v in enumerate(correctors)}
        size = len(measured)
        matrix, order = [], []
        for i, v in enumerate(measured):
            label = self.measurements[v].label
            adjacent = sum(1 << j for j, w in enumerate(correctors) if adjacency[pos[v]] & (1 << pos[w]))
            own = 1 << indices[v] if v in indices else 0
            row = adjacent if label in {"XY", "X", "Y"} else own
            if label == "Y":
                row ^= own
            matrix.append(row | (1 << (size + i)))
            order.append(own if label == "XY" else adjacent ^ own if label == "XZ" else adjacent if label == "YZ" else 0)
        for col in range(size):
            pivot = next((j for j in range(col, size) if matrix[j] & (1 << col)), None)
            if pivot is None:
                return None
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            for j in range(size):
                if j != col and matrix[j] & (1 << col):
                    matrix[j] ^= matrix[col]
        inverse = [row >> size for row in matrix]
        successors = [0] * size
        for v, demand in enumerate(order):
            product = 0
            for j in _bits(demand):
                product ^= inverse[j]
            for u in _bits(product):
                successors[u] |= 1 << v
        active = (1 << size) - 1
        layers = dict.fromkeys(self.outputs, 0)
        depth = 0
        while active:
            ready = [i for i in _bits(active) if not successors[i] & active]
            if not ready:
                return None
            depth += 1
            for i in ready:
                layers[measured[i]] = depth
                active ^= 1 << i
        corrections = {v: {w for j, w in enumerate(correctors) if inverse[j] & (1 << i)}
                       for i, v in enumerate(measured)}
        return {v: depth - layer for v, layer in layers.items()}, corrections


def _adjacency(pattern: PauliPattern) -> tuple[dict[Any, int], list[int]]:
    positions = {v: i for i, v in enumerate(pattern.vertices)}
    adjacency = [0] * len(positions)
    for u, v in pattern.edges:
        adjacency[positions[u]] |= 1 << positions[v]
        adjacency[positions[v]] |= 1 << positions[u]
    return positions, adjacency


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
    """Images C X_q C† and C Z_q C†; a unitary specified up to global phase."""

    x_images: tuple[PauliString, ...]
    z_images: tuple[PauliString, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_images", tuple(self.x_images))
        object.__setattr__(self, "z_images", tuple(self.z_images))
        n = len(self.x_images)
        if len(self.z_images) != n or any(p.n_qubits != n for p in self.x_images + self.z_images):
            raise PauliExtractionError("Clifford generator images have incompatible dimensions")
        for i in range(n):
            for j in range(n):
                if (
                    not self.x_images[i].commutes(self.x_images[j])
                    or not self.z_images[i].commutes(self.z_images[j])
                    or self.x_images[i].commutes(self.z_images[j]) != (i != j)
                ):
                    raise PauliExtractionError(
                        "Clifford images violate the Pauli commutation relations"
                    )

    @property
    def n_qubits(self) -> int:
        return len(self.x_images)

    def to_circuit(self) -> Circuit:
        """Synthesise signed generator images by symplectic elimination."""
        return _synthesise_clifford(self)

    def to_matrix(self, *, max_qubits: int = 8) -> np.ndarray:
        """Small-case verification only; choose an arbitrary global phase."""
        n = self.n_qubits
        _dense_limit(n, max_qubits)
        dim = 1 << n
        zs = [p.to_matrix(max_qubits=max_qubits) for p in self.z_images]
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
            raise PauliExtractionError("inconsistent Clifford stabilizers")
        columns = []
        for basis in range(dim):
            column = zero.copy()
            for q, x in enumerate(xs):
                if basis & (1 << (n - 1 - q)):
                    column = x @ column
            columns.append(column)
        return np.column_stack(columns)


@dataclass(frozen=True, slots=True)
class PauliExponential:
    """exp(-i*pi*angle*pauli/2), with angle in multiples of pi."""

    pauli: PauliString
    angle: Fraction
    vertex: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "angle", _phase(self.angle))


@dataclass(frozen=True)
class PauliCircuit:
    """An extracted Clifford map and ordered Pauli exponentials.

    Apply ``clifford`` first, then ``rotations`` in tuple/time order. Extraction
    preserves this representation without constructing elementary gates.
    ``flow_layers`` retains the witness's output-first partial order, and each
    rotation retains its source vertex for tracing. Call ``to_circuit()``
    explicitly for a basic gate decomposition without ancillas.
    """

    clifford: CliffordMap
    rotations: tuple[PauliExponential, ...]
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    flow_layers: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotations", tuple(self.rotations))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "flow_layers", tuple(tuple(layer) for layer in self.flow_layers))
        if any(r.pauli.n_qubits != self.n_qubits for r in self.rotations):
            raise ValueError("rotation and Clifford register sizes differ")

    @property
    def n_qubits(self) -> int:
        return self.clifford.n_qubits

    def to_circuit(self) -> Circuit:
        """Explicitly convert to elementary gates, up to a global scalar.

        This reference conversion synthesises the Clifford map and each
        Pauli exponential independently, using local basis changes and CNOT
        ladders. It does not optimize across exponentials.
        """
        circuit = self.clifford.to_circuit()
        for rotation in self.rotations:
            _append_rotation(circuit, rotation)
        return circuit

    def to_matrix(self, *, max_qubits: int = 8) -> np.ndarray:
        result = self.clifford.to_matrix(max_qubits=max_qubits)
        for rotation in self.rotations:
            theta = (np.pi * float(rotation.angle)) / 2
            result = np.cos(theta) * result - 1j * np.sin(theta) * (
                rotation.pauli.to_matrix(max_qubits=max_qubits) @ result
            )
        return result

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
        raise PauliExtractionError("anticommuting stabilizer row product")
    return x, z, a[2] ^ b[2] ^ ((phase // 2) & 1)


def _odd(s: int, adjacency: list[int]) -> int:
    value = 0
    for v in _bits(s):
        value ^= adjacency[v]
    return value


def _checked_corrections(
    pattern: PauliPattern, flow: Flow, positions: dict[Any, int], adjacency: list[int],
) -> tuple[dict[Any, int], dict[Any, int], tuple[tuple[Any, ...], ...]]:
    """Validate the full focused witness, including incomparable vertices."""
    forward, correction_sets = flow
    measured = set(pattern.measurements)
    if set(forward) != set(pattern.vertices) or set(correction_sets) != measured:
        raise PauliExtractionError("flow domains do not match the pattern")
    if any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in forward.values()):
        raise PauliExtractionError("flow layers must be nonnegative integers")
    last = max(forward.values(), default=0)
    depth = {v: last - layer + (not pattern.outputs) for v, layer in forward.items()}
    if {v for v, d in depth.items() if d == 0} != set(pattern.outputs):
        raise PauliExtractionError("flow outputs must be exactly layer zero")
    labels = {v: m.label for v, m in pattern.measurements.items()}
    masks = {label: sum(1 << positions[v] for v in labels if labels[v] == label)
             for label in _LABELS}
    outputs = sum(1 << positions[v] for v in pattern.outputs)
    planar = masks["XY"] | masks["XZ"] | masks["YZ"]
    allowed = set(pattern.vertices) - set(pattern.inputs)
    corrections, odds = {}, {}
    for u in pattern.measurements:
        if not set(correction_sets[u]) <= allowed:
            raise PauliExtractionError("correction sets contain an input or an unknown vertex")
        s = sum(1 << positions[v] for v in correction_sets[u])
        odd = _odd(s, adjacency)
        bit = 1 << positions[u]
        own, own_odd = bool(s & bit), bool(odd & bit)
        valid = {"XY": not own and own_odd, "XZ": own and own_odd,
                 "YZ": own and not own_odd, "X": own_odd,
                 "Y": own != own_odd, "Z": own}[labels[u]]
        focus_bad = ((s & (masks["XZ"] | masks["YZ"] | masks["Z"]))
                     | (odd & (masks["XY"] | masks["X"]))
                     | ((s ^ odd) & masks["Y"])) & ~bit
        successors = ((s & (planar | masks["Z"] | outputs))
                      | (odd & (planar | masks["X"] | outputs))
                      | ((s ^ odd) & masks["Y"])) & ~bit
        if not valid or focus_bad or any(
            depth[u] <= depth[pattern.vertices[v]] for v in _bits(successors)
        ):
            raise PauliExtractionError(f"supplied flow is not a focused Pauli flow at vertex {u}")
        corrections[u], odds[u] = s, odd
    layers = (tuple(pattern.outputs),) + tuple(
        tuple(v for v in pattern.vertices if depth[v] == d)
        for d in sorted(set(depth.values()) - {0})
    )
    return corrections, odds, layers


def _residual_clifford(
    pattern: PauliPattern, positions: dict[Any, int], adjacency: list[int],
) -> CliffordMap:
    """Project the Clifford Choi stabilizers, updating only affected rows."""
    g = pattern
    n, q = len(g.vertices), len(g.inputs)
    input_wire = {v: j for j, v in enumerate(g.inputs)}
    rows: list[_Row] = []
    for v in g.vertices:
        i = positions[v]
        x, z = 1 << i, adjacency[i]
        if v in input_wire:
            ref = 1 << (n + input_wire[v])
            rows.extend(((x | ref, z, 0), (0, (1 << i) | ref, 0)))
        else:
            rows.append((x, z, 0))
    physical = (1 << n) - 1
    xs, zs = [0] * n, [0] * n
    for j, row in enumerate(rows):
        for i in _bits(row[0] & physical):
            xs[i] |= 1 << j
        for i in _bits(row[1] & physical):
            zs[i] |= 1 << j
    active = (1 << len(rows)) - 1
    for v, measurement in pattern.measurements.items():
        i = positions[v]
        bit = 1 << i
        label = measurement.label
        axis = ("X" if label == "XY" else "Z") if label in _PLANES else label
        anti = (zs[i] if axis == "X" else xs[i] if axis == "Z" else xs[i] ^ zs[i]) & active
        if not anti:
            raise PauliExtractionError(
                "residual measurement has no pivot; pattern is not unitary"
            )
        pivot_bit = anti & -anti
        source = rows[pivot_bit.bit_length() - 1]
        active ^= pivot_bit
        targets = anti ^ pivot_bit
        for j in _bits(targets):
            rows[j] = _multiply(rows[j], source)
        # The same source is added to all targets: update each incidence
        # column once with a packed XOR, rather than once per target row.
        for k in _bits(source[0] & physical):
            xs[k] ^= targets
        for k in _bits(source[1] & physical):
            zs[k] ^= targets
        # All live rows now commute with the measured Pauli. Their local
        # factor is I or that Pauli; projection removes it and retains its
        # signed outcome. Dead row bits in the columns are masked by active.
        local = (xs[i] | zs[i]) & active
        negative = int(label not in _PLANES and measurement.angle == 1)
        for j in _bits(local):
            x, z, sign = rows[j]
            rows[j] = x & ~bit, z & ~bit, sign ^ negative
        xs[i] = zs[i] = 0
    rows = [rows[j] for j in _bits(active)]
    if len(rows) != 2 * q:
        raise PauliExtractionError("residual Choi state has an unexpected rank")
    for col in range(2 * q):
        component, wire = divmod(col, q)
        bit = 1 << (n + wire)
        pivot = next((j for j in range(col, len(rows)) if rows[j][component] & bit), None)
        if pivot is None:
            raise PauliExtractionError("residual Clifford is not a unitary map")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for j in range(len(rows)):
            if j != col and rows[j][component] & bit:
                rows[j] = _multiply(rows[j], rows[col])

    def image(row: _Row) -> PauliString:
        x = sum(1 << j for j, v in enumerate(g.outputs) if row[0] & (1 << positions[v]))
        z = sum(1 << j for j, v in enumerate(g.outputs) if row[1] & (1 << positions[v]))
        return PauliString(q, x, z, -1 if row[2] else 1)

    return CliffordMap(
        tuple(image(row) for row in rows[:q]), tuple(image(row) for row in rows[q:])
    )


def _hadamard(pauli: PauliString, mask: int) -> PauliString:
    swap = (pauli.x ^ pauli.z) & mask
    sign = -pauli.sign if (pauli.x & pauli.z & mask).bit_count() & 1 else pauli.sign
    return PauliString(pauli.n_qubits, pauli.x ^ swap, pauli.z ^ swap, sign)


def extract_pauli(
    pattern: PauliPattern | BaseGraph, *, flow: Flow | None = None,
    infer_pauli: bool = True,
) -> PauliCircuit:
    """Return a :class:`PauliCircuit` following Simmons's extraction.

    The result retains a Clifford map and ordered Pauli exponentials, without
    synthesising elementary gates. Use :meth:`PauliCircuit.to_circuit` for an
    explicit basic gate decomposition.

    A graph input must be graph-like; it is not mutated. A supplied ``flow``
    is the ``(layers, corrections)`` pair returned by :func:`pyzx.flow.pauli_flow`
    on the same graph, or by :meth:`PauliPattern.find_flow` for explicit labels.
    The witness must be focused, forward, and include correction sets.
    ``infer_pauli`` applies only to graph conversion; use False with a gflow
    witness. Grounded graphs, unequal boundary counts and symbolic phases are
    rejected. Results omit the global scalar.
    """
    if isinstance(pattern, BaseGraph):
        graph = pattern
        pattern = PauliPattern.from_graph(graph, infer_pauli=infer_pauli)
        if len(pattern.inputs) != len(pattern.outputs):
            raise PauliExtractionError("Pauli extraction requires equal input/output counts")
        if flow is None:
            flow = pauli_flow(graph) if infer_pauli else gflow(graph)
            if flow is None:
                raise PauliExtractionError("the graph has no Pauli flow")
        if flow is not None and pattern._direct_wires:
            flow = (dict(flow[0]), flow[1])
            flow[0].update(dict.fromkeys(pattern._direct_wires, max(flow[0].values(), default=0)))
    if not isinstance(pattern, PauliPattern):
        raise TypeError("extract_pauli expects a PauliPattern or a PyZX graph")
    g = pattern
    if len(g.inputs) != len(g.outputs):
        raise PauliExtractionError("Pauli extraction requires equal input/output counts")
    if flow is None:
        flow = pattern.find_flow()
        if flow is None:
            raise PauliExtractionError("the pattern has no Pauli flow")
    positions, adjacency = _adjacency(pattern)
    corrections, odds, layers = _checked_corrections(pattern, flow, positions, adjacency)
    output_positions = [positions[v] for v in g.outputs]
    negative_axes = sum(
        1 << positions[v]
        for v, m in pattern.measurements.items()
        if m.label not in _PLANES and m.angle == 1
    )
    rotations = []
    for layer in reversed(layers[1:]):
        for v in layer:
            measurement = pattern.measurements[v]
            if measurement.label not in _PLANES or measurement.angle == 0:
                continue
            s, odd = corrections[v], odds[v]
            # Simmons (33)--(35): induced edges, pairs of Ys, absorbed negatives.
            edge_count = sum((adjacency[i] & s & ((1 << i) - 1)).bit_count() for i in _bits(s))
            negative = (
                edge_count + (s & odd).bit_count() // 2 + ((s | odd) & negative_axes).bit_count()
            ) & 1
            x = sum(1 << j for j, i in enumerate(output_positions) if s & (1 << i))
            z = sum(1 << j for j, i in enumerate(output_positions) if odd & (1 << i))
            pauli = PauliString(len(g.outputs), x, z, -1 if negative else 1)
            angle = measurement.angle if measurement.label == "YZ" else -measurement.angle
            rotations.append(PauliExponential(pauli, angle, v))
    clifford = _residual_clifford(pattern, positions, adjacency)
    xs, zs = list(clifford.x_images), list(clifford.z_images)
    for q in pattern.input_hadamards:
        xs[q], zs[q] = zs[q], xs[q]
    h_mask = sum(1 << q for q in pattern.output_hadamards)
    clifford = CliffordMap(
        tuple(_hadamard(p, h_mask) for p in xs), tuple(_hadamard(p, h_mask) for p in zs)
    )
    rotations = [PauliExponential(_hadamard(r.pauli, h_mask), r.angle, r.vertex) for r in rotations]
    for q, angle in enumerate(pattern.output_phases):
        if angle != 0:
            rotations.append(
                PauliExponential(_hadamard(PauliString(len(g.outputs), 0, 1 << q), h_mask), angle)
            )
    return PauliCircuit(clifford, tuple(rotations), g.inputs, g.outputs, layers)


def extract_circuit_pauli(
    g: PauliPattern | BaseGraph, *, flow: Flow | None = None,
    infer_pauli: bool = True,
) -> PauliCircuit:
    """Return a :class:`PauliCircuit`, as :func:`extract_pauli` does.

    The Clifford map and Pauli exponentials are preserved; no elementary
    gates are synthesised. Call ``result.to_circuit()`` explicitly for a
    basic gate decomposition. See :func:`extract_pauli` for domain, phase
    and witness conventions.
    """
    return extract_pauli(g, flow=flow, infer_pauli=infer_pauli)


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
            rows[j] = x, z, int(negative)

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


def _append_rotation(circuit: Circuit, rotation: PauliExponential) -> None:
    """Change local bases, compute parity, rotate, then uncompute."""
    pauli = rotation.pauli
    targets = list(_bits(pauli.x | pauli.z))
    if not targets or rotation.angle == 0:
        return  # A rotation about the identity contributes only a global phase.
    basis = Circuit(circuit.qubits)
    for q in targets:
        if pauli.x & (1 << q):
            if pauli.z & (1 << q):
                basis.add_gate("S", q, adjoint=True)
            basis.add_gate("HAD", q)
    circuit.add_circuit(basis)
    circuit.gates.extend(ParityPhase(rotation.angle * pauli.sign, *targets).to_basic_gates())
    circuit.add_circuit(basis.adjoint())
