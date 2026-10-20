Pauli-flow circuit extraction
=============================

``pyzx.extract_pauli`` implements the construction in Will Simmons,
`Relating Measurement Patterns to Circuits via Pauli Flow
<https://doi.org/10.4204/EPTCS.343.4>`_ (2021), Lemma C.2 and Theorem C.12.
It returns a :class:`~pyzx.pauli_extract.PauliCircuit` containing a Clifford
map and ordered rotations about Pauli tensors. Extraction preserves this
representation without synthesising elementary gates.
``pyzx.extract_circuit_pauli`` is an equivalent entry point returning the same
type. Both are separate from the frontier-based
:func:`pyzx.extract.extract_circuit`, which returns an ordinary circuit.

Graph inputs
------------

The input must be a graph-like diagram: Z-spiders joined by Hadamard edges,
with degree-one input/output boundary vertices. The number of inputs and
outputs must agree. Ground vertices, symbolic phases, self-loops and parallel
edges are rejected. The graph is not modified. Input/output order, boundary
Hadamards and output phases are preserved, including direct input-output wires.
The extracted map agrees up to a nonzero global scalar, which is not tracked.
No ancillary wires are added.

.. code-block:: python

    import pyzx as zx

    g = circuit.to_graph()
    zx.to_graph_like(g)
    result = zx.extract_pauli(g)

A measured Z-spider with phase ``a`` represents an XY measurement at angle
``-a`` in Simmons's conventions. Exact Clifford angles are interpreted as
signed X/Y measurements by default. This agrees with the labels used by
:func:`pyzx.flow.pauli_flow`; rational angles are never rounded. Use
``infer_pauli=False`` to retain the XY planes and use gflow instead.

Reusing a flow witness
----------------------

.. code-block:: python

    from pyzx.flow import pauli_flow

    flow = pauli_flow(g)
    if flow is None:
        raise ValueError("No Pauli flow")
    result = zx.extract_pauli(g, flow=flow)

``flow`` is the normal forward ``(layers, corrections)`` pair, with correction
sets. It must be focused and refer to the same graph and measurement labels.
The extractor verifies its domain, input exclusion, local Pauli conditions,
focusing and layer order, without invoking flow finding again. Reversed flows
and results obtained with ``layers_only=True`` are not extraction witnesses.
For a gflow witness, also pass ``infer_pauli=False``.

Intermediate representation and explicit labels
-----------------------------------------------

.. code-block:: python

    result = zx.extract_pauli(g, flow=flow)
    clifford = result.clifford
    rotations = result.rotations
    assert isinstance(result, zx.PauliCircuit)

The Clifford map stores signed generator images ``C X_q C†`` and ``C Z_q C†``.
Each :class:`~pyzx.pauli_extract.PauliExponential` represents
``exp(-i*pi*angle*P/2)``, where ``P`` is a signed Hermitian Pauli tensor.
Apply the Clifford first, followed by rotations in tuple order. Rotations
retain their source vertex. ``flow_layers`` is stored output-first, with
outputs in its first tuple, independently of the numbering in the supplied
PyZX witness. The capped ``to_matrix()`` methods are verification helpers;
extraction and gate synthesis do not construct dense matrices.

Optional elementary-gate conversion
-----------------------------------

.. code-block:: python

    circuit = result.to_circuit()

This explicit reference conversion synthesises the Clifford map and each
Pauli exponential independently, using local basis changes and CNOT ladders.
It does not optimize across exponentials. Retain ``result.clifford`` and
``result.rotations`` to pass the structured computation to a joint Pauli
synthesizer instead.

Explicit measurement labels
----------------------------

:class:`~pyzx.pauli_extract.PauliPattern` can describe an explicit labelled
open graph with XY, XZ, YZ, X, Y and Z measurements. Unlike a ZX graph, its
inputs and outputs are qubit vertices, and may overlap. For example:

.. code-block:: python

    from fractions import Fraction
    from pyzx.pauli_extract import Measurement, PauliPattern

    pattern = PauliPattern(
        vertices=(0, 1), edges=((0, 1),), inputs=(0,), outputs=(1,),
        measurements={0: Measurement("XY", Fraction(1, 3))},
    )
    flow = pattern.find_flow()
    result = zx.extract_pauli(pattern, flow=flow)

``Measurement`` follows Simmons's positive-outcome ket conventions:
XY uses ``(|0> + exp(i*pi*a)|1>)/sqrt(2)``, XZ uses
``cos(pi*a/2)|0> + sin(pi*a/2)|1>``, and YZ replaces the sine term by
``i*sin(pi*a/2)|1>``. Axis labels use angle 0 or 1 for the positive or
negative eigenstate. All angles are rational multiples of pi.

For explicit patterns, ``find_flow()`` uses the published Mitosek--Backens
M/N equations (DOI `10.1088/1751-8121/ae2999
<https://doi.org/10.1088/1751-8121/ae2999>`_). Equal input/output counts make
M square: invert M over GF(2), then check that NC is acyclic. This helper
does not extend the public graph-based flow finders to additional planes.

Implementation
--------------

Pauli supports and stabilizer incidence columns are packed into Python
integers. The output rotations use the focused correction sets. The
Clifford remainder is reconstructed by projecting a Choi stabilizer tableau
after removing the planar rotations. Incidence columns locate affected rows,
avoiding a scan of every row at each measurement. Its input-reference block
is then reduced to recover the Clifford-generator images. This avoids another
flow solve on an input-extended graph.

The packed extraction core and the square flow helper use :math:`O(N^3)` bit work
and :math:`O(N^2)` stored bits, in addition to the output and exact-angle arithmetic.
Optional gate synthesis is a separate step: symplectic elimination produces the
Clifford gates, and local basis changes and CNOT ladders produce the Pauli
rotations. There is no claim that this strategy reduces gate count or is
uniformly faster than frontier-based extraction.

API
---

.. automodule:: pyzx.pauli_extract
   :members:
