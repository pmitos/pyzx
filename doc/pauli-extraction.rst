Pauli-flow circuit extraction
=============================

``pyzx.extract_pauli`` follows the decomposition into a Clifford isometry
and Pauli rotations in Will Simmons,
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
with degree-one input/output boundary vertices. There must be at least as
many outputs as inputs. Ground vertices, symbolic phases, self-loops and parallel
edges are rejected. The graph is not modified. Input/output order, boundary
Hadamards and output phases are preserved, including direct input-output wires
and multiple outputs attached to a shared spider.
The extracted map agrees up to a nonzero global scalar, which is not tracked.
Extra outputs represent prepared qubits; no wires beyond the output register
are added. Patterns with no inputs are supported as state preparations.

.. code-block:: python

    import pyzx as zx

    g = circuit.to_graph()
    zx.to_graph_like(g)
    result = zx.extract_pauli(g)

A measured Z-spider with phase ``a`` represents an XY measurement at angle
``-a`` in Simmons's conventions. Exact Clifford angles are interpreted as
signed X/Y measurements by default. This agrees with the labels used by
:func:`pyzx.flow.pauli_flow`; rational angles are never rounded. Use
``infer_pauli=False`` to keep all measured spiders as XY and use gflow instead.
The extractor takes a ZX diagram. Other measurement planes can appear in
the diagram through ZX encodings such as phase gadgets.

Reusing a flow witness
----------------------

.. code-block:: python

    from pyzx.flow import pauli_flow

    flow = pauli_flow(g)
    if flow is None:
        raise ValueError("No Pauli flow")
    result = zx.extract_pauli(g, flow=flow)

A supplied ``flow`` is assumed to be a valid, focused, forward
``(layers, corrections)`` witness with correction sets for the same graph and measurement labels. The
extractor checks its structure and reuses the witness. The caller is responsible for the Pauli-flow conditions,
including focusing and precedence. Reversed flows and results obtained with
``layers_only=True`` are not extraction witnesses.

With ``flow=None`` (the default), the extractor calls
:func:`pyzx.flow.pauli_flow`. With ``infer_pauli=False`` it calls
:func:`pyzx.flow.gflow` instead. Also pass ``infer_pauli=False`` when supplying
a gflow witness.

Intermediate representation
---------------------------

.. code-block:: python

    result = zx.extract_pauli(g, flow=flow)
    clifford = result.clifford
    rotations = result.rotations
    assert isinstance(result, zx.PauliCircuit)

The Clifford map stores signed logical Pauli images satisfying ``X'_q C = C X_q``
and ``Z'_q C = C Z_q``. For a unitary these are the usual ``C X_q C†`` and
``C Z_q C†``. With more outputs than inputs, ``clifford.stabilizers`` additionally
contains one independent signed output stabilizer per extra output, satisfying
``S C = C``. Together these specify the Clifford isometry up to global phase,
without choosing a unitary completion. ``n_inputs`` and ``n_outputs`` give its
dimensions; ``n_qubits`` denotes the output register size. A state preparation
has no logical images and is specified entirely by its output stabilizers.

Each rotation is a :class:`~pyzx.circuit.PauliExponential` gate representing
``exp(-i*pi*angle*P/2)``. Its ``pauli`` attribute is a signed string of
``I``, ``X``, ``Y`` and ``Z``, with wire zero leftmost, and ``angle`` is in
units of pi. Extracted gates use the default ``gadget_phase=False`` convention.
They can be added directly to an ordinary :class:`~pyzx.circuit.Circuit`.
Simmons uses a positive imaginary exponent for XY measurements, so the
extractor negates the XY measurement angle before reducing it modulo two.
The Pauli retains the sign of the primary extraction string.

Apply the Clifford first, followed by rotations in tuple order.
``rotation_vertices`` is an aligned tuple of source vertices, with ``None``
for rotations from output phases. For example, iterate over
``zip(result.rotations, result.rotation_vertices)`` to trace each gate.
``flow_layers`` is stored output-first, with
outputs in its first tuple, independently of the numbering in the supplied
PyZX witness. The capped ``to_matrix()`` methods are verification helpers;
extraction and gate synthesis do not construct dense matrices.

Optional elementary-gate conversion
-----------------------------------

.. code-block:: python

    circuit = result.to_circuit()

This explicit reference conversion synthesises the Clifford map and calls
``to_basic_gates()`` on each Pauli exponential, using local basis changes and
CNOT ladders.
For an isometry, it first completes the tableau to a Clifford unitary, with
the extra X images given by the output stabilizers. Extra wires are prepared
in :math:`|+\rangle` using :class:`~pyzx.circuit.gates.InitAncilla` gates.
The returned ``Circuit.qubits`` counts the input wires; the initialization
gates add wires numbered from ``n_inputs`` through ``n_outputs - 1``.
It does not optimize across exponentials. Retain ``result.clifford`` and
``result.rotations`` to pass the structured computation to a joint Pauli
synthesizer instead.

``circuit.to_qasm(version=2)`` and ``circuit.to_qasm(version=3)`` declare
the full output register and prepare each added wire with a reset followed
by a Hadamard. The first ``n_inputs`` wires carry the original inputs.

Implementation
--------------

The Pauli rotations use the extraction strings derived from a focused Pauli
flow in Lemma C.2. In the proof of Theorem C.12, Simmons obtains the logical
Pauli images from extraction strings, using input extensions for the X
images. The free output stabilizers come from focused sets.

This implementation reconstructs the residual Clifford isometry by projecting
a Choi stabilizer tableau. It sets the XY measurement angles to zero and
keeps the signed Pauli measurements. Elimination on the input-reference
columns yields the logical X and Z images, and the remaining rows give the
free output stabilizers. This computes the tableau without a flow solve on
an input-extended graph.

Pauli supports and stabilizer incidence columns are packed into Python
integers during extraction. The output rotation supports are converted to
signed strings when constructing the gates. The incidence columns locate
the rows affected by each measurement.
Shared output spiders are separated internally, and a supplied witness is
extended to account for the added vertices.

Focused flow determines the measured part of :math:`\operatorname{Odd}(p(v))` as
:math:`\{v\} \mathbin{\triangle} (p(v) \cap Y_{\rm meas})`, where
:math:`Y_{\rm meas}` is the set of Y-measured vertices. The extractor computes
the output part by XORing packed output-neighbour masks in the same pass
that counts edges induced by :math:`p(v)` for the rotation sign.

The packed extraction core uses :math:`O(N^3)` bit work and :math:`O(N^2)`
stored bits, in addition to the output and exact-angle arithmetic.
Optional gate synthesis is a separate step: symplectic elimination produces the
Clifford gates, and local basis changes and CNOT ladders produce the Pauli
rotations. There is no claim that this strategy reduces gate count or is
uniformly faster than frontier-based extraction.

API
---

.. automodule:: pyzx.pauli_extract
   :members:
