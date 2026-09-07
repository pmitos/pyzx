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

from fractions import Fraction
from typing import Dict, Set, Tuple, Optional, Iterator

from .linalg import Mat2
from .graph.base import BaseGraph, VT, ET
from .utils import phase_is_clifford, phase_is_pauli, vertex_is_zx


def gflow(
    g: BaseGraph[VT, ET], focus: bool=False, reverse: bool=False, pauli: bool=False,
    *, method: str="cubic"
) -> Optional[Tuple[Dict[VT, int], Dict[VT, Set[VT]]]]:
    r"""Find gflow or Pauli flow for {XY, X, Y} measurements.

    :param g: A graph-like ZX diagram.
    :param focus: Require focused corrections, including constraints on grounds.
    :param reverse: Reverse the roles of inputs and outputs.
    :param pauli: Interpret Pauli phases as X or Y measurements.
    :param method: ``cubic`` (default) or the retained ``legacy`` finder.

    The cubic method returns focused corrections even when ``focus=False``.
    In that mode grounds are treated as outputs. With ``focus=True`` and
    non-output grounds, the legacy finder preserves PyZX's ground constraints.
    Corrections and layer numbers need not be identical between methods.
    Ordinary order runs from smaller to larger
    layer numbers; ``reverse=True`` returns the opposite numbering convention.

    This ports mbqcflow's Mitosek--Backens right-inverse/kernel algorithm
    (https://arxiv.org/abs/2410.23439), restricted to XY, X and Y. M is the
    flow-demand matrix (adjacency with a Y diagonal); N selects XY correction
    coordinates. For square M, compute C = M^-1 and check that NC is a DAG.
    Otherwise compute a right inverse C0 and kernel basis K, then find P with
    N(C0 + KP) acyclic using the maintained system [NK | NC0 | I].
    Packed Python integers give O(n^3) bit operations and O(n^2) bits of storage
    without new dependencies. The focused-ground fallback retains legacy
    complexity.
    """
    if method == "legacy":
        return _gflow_legacy(g, focus=focus, reverse=reverse, pauli=pauli)
    if method != "cubic":
        raise ValueError("Unknown flow method: " + method)

    vertices = [v for v in g.vertices() if vertex_is_zx(g.type(v))]
    vertex_set = set(vertices)
    inputs = {v for b in g.inputs() for v in g.neighbors(b) if v in vertex_set}
    outputs = {v for b in g.outputs() for v in g.neighbors(b) if v in vertex_set}
    if reverse:
        inputs, outputs = outputs, inputs
    processed = outputs | (g.grounds() & vertex_set)
    if focus and processed != outputs:
        # Grounds have homogeneous demand constraints but no unit RHS of their
        # own. This is not the MC = I problem solved by mbqcflow's algorithm.
        return _gflow_legacy(g, focus=focus, reverse=reverse, pauli=pauli)
    paulis = set()
    ys = set()
    if pauli:
        for v in vertices:
            phase = g.phase(v) % 2
            if phase_is_pauli(phase):
                paulis.add(v)
            elif phase_is_clifford(phase):
                paulis.add(v)
                ys.add(v)

    rows = [v for v in vertices if v not in processed]
    columns = [v for v in vertices if v not in inputs]
    column_index = {v: j for j, v in enumerate(columns)}
    demand = []
    for v in rows:
        bits = 0
        for w in g.neighbors(v):
            if w in column_index:
                bits |= 1 << column_index[w]
        if v in ys and v in column_index:
            bits |= 1 << column_index[v]
        demand.append(bits)
    # N has at most one entry per row for PyZX's XY/X/Y measurements.
    order_columns = [column_index.get(v, -1) if v not in paulis else -1 for v in rows]
    result = _dag_right_inverse(demand, order_columns, len(columns))
    if result is None:
        return None
    correction_matrix, batches = result
    layers = {v: 0 for v in processed}
    for depth, batch in enumerate(batches, 1):
        for i in batch:
            layers[rows[i]] = depth
    corrections: Dict[VT, Set[VT]] = {v: set() for v in rows}
    for j, bits in enumerate(correction_matrix):
        for i in _set_bits(bits):
            corrections[rows[i]].add(columns[j])
    depth = len(batches)
    return ({v: layer if reverse else depth - layer for v, layer in layers.items()},
            corrections)


def _set_bits(bits: int) -> Iterator[int]:
    """Yield the indices of nonzero entries in a packed binary row."""
    while bits:
        bit = bits & -bits
        yield bit.bit_length() - 1
        bits ^= bit


def _right_inverse(
    matrix: list[int], column_count: int
) -> Optional[Tuple[list[int], list[int], list[int]]]:
    """Reduce [M | I], returning C0, reduced rows and pivot columns.

    Free variables of C0 are zero. Retain the reduction so the rectangular
    path can construct ker(M) without another elimination.
    """
    row_count = len(matrix)
    if row_count > column_count:
        return None
    reduced = [row | (1 << (column_count + i)) for i, row in enumerate(matrix)]
    pivots: list[int] = []
    for j in range(column_count):
        rank = len(pivots)
        pivot = next((i for i in range(rank, row_count) if reduced[i] & (1 << j)), None)
        if pivot is None:
            continue
        reduced[rank], reduced[pivot] = reduced[pivot], reduced[rank]
        for i in range(row_count):
            if i != rank and reduced[i] & (1 << j):
                reduced[i] ^= reduced[rank]
        pivots.append(j)
        if len(pivots) == row_count:
            break
    if len(pivots) != row_count:
        return None
    correction = [0] * column_count
    for i, j in enumerate(pivots):
        correction[j] = reduced[i] >> column_count
    return correction, reduced, pivots


def _kernel_basis(reduced: list[int], pivots: list[int], column_count: int) -> list[int]:
    """Return packed rows of a kernel basis from the reduction of [M | I]."""
    pivot_set = set(pivots)
    free = [j for j in range(column_count) if j not in pivot_set]
    kernel = [0] * column_count
    for k, j in enumerate(free):
        kernel[j] = 1 << k
        for i, pivot in enumerate(pivots):
            if reduced[i] & (1 << j):
                kernel[pivot] |= 1 << k
    return kernel


def _dag_right_inverse(
    demand: list[int], order_columns: list[int], column_count: int
) -> Optional[Tuple[list[int], list[list[int]]]]:
    """Find MC=I with NC acyclic, using mbqcflow's square/general split.

    N is represented by its selected column in each row, or -1 for a zero row.
    All other matrices are lists of packed binary rows.
    """
    inverse = _right_inverse(demand, column_count)
    if inverse is None:
        return None
    correction, reduced, pivots = inverse
    row_count = len(demand)
    if row_count != column_count:
        kernel = _kernel_basis(reduced, pivots, column_count)
        left = [kernel[j] if j >= 0 else 0 for j in order_columns]
        middle = [correction[j] if j >= 0 else 0 for j in order_columns]
        adjustment = _find_kernel_adjustment(left, middle, column_count - row_count)
        if adjustment is None:
            return None
        # C = C0 + KP over GF(2).
        for j, row in enumerate(kernel):
            for k in _set_bits(row):
                correction[j] ^= adjustment[k]
    # In particular, square M never constructs a kernel or solves for P.
    order_product = [correction[j] if j >= 0 else 0 for j in order_columns]
    layers = _dag_layers(order_product)
    return None if layers is None else (correction, layers)


def _find_kernel_adjustment(left: list[int], middle: list[int], k: int) -> Optional[list[int]]:
    """Find P with B+AP acyclic via mbqcflow's maintained [A | B | I].

    Here A=NK and B=NC0. After a batch is solved, remove its original row
    constraints using the identity block, and restore echelon order by updating
    one row per removed constraint. Do not re-eliminate the full system.
    """
    n = len(left)
    coefficient_mask = (1 << k) - 1
    rhs_mask = (1 << n) - 1
    invariant = [a | (b << k) | (1 << (k + n + i))
                 for i, (a, b) in enumerate(zip(left, middle))]
    system = invariant.copy()
    rank = 0
    for j in range(k):
        pivot = next((i for i in range(rank, n) if system[i] & (1 << j)), None)
        if pivot is None:
            continue
        system[rank], system[pivot] = system[pivot], system[rank]
        for i in range(rank + 1, n):
            if system[i] & (1 << j):
                system[i] ^= system[rank]
        rank += 1

    adjustment = [0] * k
    remaining = rhs_mask
    while remaining:
        blocked = 0
        pivot_rows = []
        for row in system:
            coefficients = row & coefficient_mask
            if coefficients:
                pivot_rows.append((coefficients & -coefficients, coefficients, row >> k))
            else:
                blocked |= (row >> k) & rhs_mask
        solvable = remaining & ~blocked
        if not solvable:
            return None
        for i in _set_bits(solvable):
            # At most k packed parity evaluations per target; each target is
            # solved once, regardless of the number of solver layers.
            solution = 0
            for pivot_bit, coefficients, rhs in reversed(pivot_rows):
                if ((rhs >> i) & 1) ^ ((coefficients & solution).bit_count() & 1):
                    solution |= pivot_bit
            for j in _set_bits(solution):
                adjustment[j] |= 1 << i
        for i in _set_bits(solvable):
            flag = 1 << (k + n + i)
            users = [j for j, row in enumerate(system) if row & flag]
            if not users:
                continue
            replacement = users[-1]
            for j in users[:-1]:
                system[j] ^= system[replacement]
            system[replacement] ^= invariant[i]
            _restore_echelon(system, replacement, coefficient_mask)
        remaining ^= solvable
    return adjustment


def _restore_echelon(system: list[int], changed: int, coefficient_mask: int) -> None:
    """Restore echelon order after one constraint-removal row update."""
    row = system.pop(changed)
    for other in system:
        coefficients = other & coefficient_mask
        if not coefficients:
            break
        if row & (coefficients & -coefficients):
            row ^= other
    coefficients = row & coefficient_mask
    pivot = coefficients & -coefficients
    target = 0
    for other in system:
        other_coefficients = other & coefficient_mask
        if not other_coefficients:
            break
        if pivot and (other_coefficients & -other_coefficients) > pivot:
            break
        target += 1
    system.insert(target, row)


def _dag_layers(matrix: list[int]) -> Optional[list[list[int]]]:
    """Return sink-first batches of the column-to-row relation, or None.

    Count every edge once on insertion and once on removal, including diagonal
    entries (which must cause failure). There are O(n^2) edge visits.
    """
    outgoing = [0] * len(matrix)
    for row in matrix:
        for source in _set_bits(row):
            outgoing[source] += 1
    batch = [i for i, degree in enumerate(outgoing) if degree == 0]
    layers = []
    count = 0
    while batch:
        layers.append(batch)
        count += len(batch)
        following = []
        for target in batch:
            for source in _set_bits(matrix[target]):
                outgoing[source] -= 1
                if outgoing[source] == 0:
                    following.append(source)
        batch = following
    return layers if count == len(matrix) else None


def _gflow_legacy(
    g: BaseGraph[VT, ET], focus: bool=False, reverse: bool=False, pauli: bool=False
) -> Optional[Tuple[Dict[VT, int], Dict[VT, Set[VT]]]]:
    r"""Compute the gflow of a diagram in graph-like form.

    :param g: A graphlike ZX diagram.
    :param focus: Compute the focussed gflow
    :param reverse: Reverse the roles of inputs and outputs
    :param pauli: Compute the Pauli flow, restricted to {XY, X, Y} measurements

    Based on algorithm by Perdrix and Mhalla.
    See dx.doi.org/10.1007/978-3-540-70575-8_70

    Slightly extended to allow searching for Pauli flow with measurement planes {XY, X, Y}.

    Here is the pseudocode it is based on:
    ```
    input : An open graph
    output: A generalised flow

    gFlow (V,Gamma,In,Out) =
    begin
      for all v in Out do
        l(v) := 0
      end
      return gFlowaux (V,Gamma,In,Out,1)
    end

    gFlowaux (V,Gamma,In,Out,k) =
    begin
      C := {}
      for all u in V \\ Out do
        Solve in F2 : Gamma[V \\ Out, Out \\ In] * I[X] = I[{u}]
        if there is a solution X0 then
          C := C union {u}
          g(u) := X0
          l(u) := k
        end
      end
      if C = {} then
        return (Out = V,(g,l))
      else
        return gFlowaux (V, Gamma, In, Out union C, k + 1)
      end
    end
    ```
    """
    l: Dict[VT, int] = {}
    gflow: Dict[VT, Set[VT]] = {}
    ty = g.types()

    vertices: Set[VT] = set(v for v in g.vertices() if vertex_is_zx(ty[v]))
    pattern_inputs: Set[VT] = set()
    pattern_outputs: Set[VT] = set()
    pauli_x: Set[VT] = set()
    pauli_y: Set[VT] = set()

    for inp in g.inputs():
        pattern_inputs |= set(n for n in g.neighbors(inp) if vertex_is_zx(ty[n]))
    for outp in g.outputs():
        pattern_outputs |= set(n for n in g.neighbors(outp) if vertex_is_zx(ty[n]))
    
    if reverse:
        pattern_inputs, pattern_outputs = pattern_outputs, pattern_inputs

    if pauli:
        for v in vertices:
            p = g.phase(v) % 2
            if phase_is_pauli(p):
                pauli_x.add(v)
            elif phase_is_clifford(p):
                pauli_y.add(v)

    processed: Set[VT] = pattern_outputs.copy() | g.grounds()
    non_outputs = list(vertices.difference(pattern_outputs))
    zerovec = Mat2.zeros(len(non_outputs), 1)
    for v in processed:
        l[v] = 0

    k: int = 1
    while True:
        correct: Set[VT] = set()

        # a list of nodes that can currently be used in the correction set of the next node
        # Keep candidate v when its column in the current flow-demand matrix
        # is nonzero on an unprocessed row. This can arise either from a graph
        # edge or, for Pauli-Y, from the diagonal coefficient M[v, v] = 1.
        candidates = [
            v
            for v in (processed | pauli_x | pauli_y).difference(pattern_inputs)
            if focus
            or (v in pauli_y and v not in processed)
            or any(w not in processed for w in g.neighbors(v))
        ]

        if focus:
            clean = non_outputs
        else:
            clean = [v for v in vertices
                        if v not in processed and 
                        ((v in pauli_y and v in candidates)
                         or any(w in candidates for w in g.neighbors(v)))]

        # compute the "flow-demand matrix", which is essentially the bi-adjacency matrix from
        # "clean" to "candidates", which additionally relates every Y-measured node to
        # itself.
        m = Mat2([[1 if g.connected(v,w) or (v==w and v in pauli_y) else 0 
                   for v in candidates] for w in clean])

        for index, u in enumerate(clean):
            if not focus or (
                u not in processed
                and (
                    (u in pauli_y and u in candidates)
                    or any(w in candidates for w in g.neighbors(u))
                )
            ):
                vu = zerovec.copy()
                vu.data[index][0] = 1
                x = m.solve(vu)
                if x:
                    correct.add(u)
                    gflow[u] = {candidates[i] for i in range(x.rows()) if x.data[i][0]}
                    l[u] = k

        if not correct:
            if len(vertices) == len(processed):
                return {v: i if reverse else k - i - 1 for v,i in l.items()}, gflow
            return None
        else:
            processed.update(correct)
            k += 1
