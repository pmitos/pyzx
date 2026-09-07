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
from typing import Dict, Set, Tuple, Optional

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
    With grounds, that mode imposes constraints only on ungrounded non-outputs,
    matching the legacy convention. Corrections and layer numbers need not be
    identical between methods. Ordinary order runs from smaller to larger
    layer numbers; ``reverse=True`` returns the opposite numbering convention.

    This specializes the flow-demand/order-demand formulation of Mitosek and
    Backens (https://arxiv.org/abs/2410.23439). Here M is adjacency with a Y
    diagonal, and N selects only XY correction coordinates. Thus outputs,
    grounds and Pauli vertices supply correction columns immediately, while
    an XY column becomes available after its vertex is solved.

    Maintain one column basis of M and residuals for all unit right-hand sides.
    Each column is inserted once and each new pivot updates every unsolved RHS
    once. There are O(n^2) packed-vector operations on O(n)-bit integers, giving
    O(n^3) bit operations and O(n^2) bits of storage. This is an incremental
    column-basis specialization, not the general M/N kernel implementation.
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

    rows = [v for v in vertices if v not in (outputs if focus else processed)]
    row_index = {v: i for i, v in enumerate(rows)}
    columns = [v for v in vertices if v not in inputs]
    column_index = {v: j for j, v in enumerate(columns)}
    demand = []
    for v in columns:
        bits = 0
        for w in g.neighbors(v):
            if w in row_index:
                bits |= 1 << row_index[w]
        if v in ys and v in row_index:
            bits |= 1 << row_index[v]
        demand.append(bits)

    residual = [1 << i for i in range(len(rows))]
    solutions = [0] * len(rows)
    active = [i for i, v in enumerate(rows) if v not in processed]
    # Each entry is (pivot bit, transformed M column, correction coordinates).
    # Later basis vectors have zero entries at every earlier pivot.
    basis: list[tuple[int, int, int]] = []
    pending = [column_index[v] for v in columns if v in processed or v in paulis]
    inserted = set(pending)
    layers = {v: 0 for v in processed}
    corrections: Dict[VT, Set[VT]] = {}
    depth = 0
    while active:
        for j in pending:
            vector, combination = demand[j], 1 << j
            for pivot, column, coordinates in basis:
                if vector & pivot:
                    vector ^= column
                    combination ^= coordinates
            if not vector:
                continue
            pivot = vector & -vector
            basis.append((pivot, vector, combination))
            for i in active:
                if residual[i] & pivot:
                    residual[i] ^= vector
                    solutions[i] ^= combination

        solved = [i for i in active if not residual[i]]
        if not solved:
            return None
        depth += 1
        pending = []
        for i in solved:
            v = rows[i]
            layers[v] = depth
            correction = set()
            bits = solutions[i]
            while bits:
                bit = bits & -bits
                correction.add(columns[bit.bit_length() - 1])
                bits ^= bit
            corrections[v] = correction
            if v in column_index and column_index[v] not in inserted:
                j = column_index[v]
                inserted.add(j)
                pending.append(j)
        active = [i for i in active if residual[i]]

    return ({v: layer if reverse else depth - layer for v, layer in layers.items()},
            corrections)


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
