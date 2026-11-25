Clifford tableaux and isometries
================================

``pyzx.tableau`` provides a compact representation of Clifford maps, including
state preparations and isometries with more outputs than inputs. It stores
signed Pauli generator images, not a dense matrix or a gate decomposition.
The public classes are also available as ``pyzx.PauliString``,
``pyzx.CliffordMap`` and ``pyzx.CliffordMapError``.

This is an initial representation and reference-conversion API. Circuit
construction accepts only the basic Clifford gates listed below. Synthesis
uses unitary completion and symplectic elimination; it does not optimize gate
count, depth, connectivity or the choice of completion. There is no claim of
optimal synthesis. General tableau composition, inversion, measurement
simulation and file-format adapters are outside this initial API.

Signed Pauli strings
--------------------

``PauliString(n_qubits, x, z, sign=1)`` stores the X and Z supports as Python
integers. Bit ``q`` refers to wire ``q``. The local bit pairs ``(x, z)`` encode
``I = (0, 0)``, ``X = (1, 0)``, ``Z = (0, 1)`` and ``Y = (1, 1)``.
The sign is ``+1`` or ``-1`` and multiplies this Hermitian tensor. For example,
``PauliString(3, 0b001, 0b100, -1)`` represents ``-XIZ``. Displayed strings list
wire zero on the left. These are Hermitian Paulis; there is no public
general-purpose multiplication API for complex Pauli phases.

``commutes(other)`` evaluates the binary symplectic commutation condition.
``to_matrix(max_qubits=8)`` is an explicitly capped small-instance helper.

Clifford isometry convention
----------------------------

For a map ``C`` with ``n`` inputs and ``m`` outputs, a ``CliffordMap`` stores:

* ``x_images``: ``n`` signed output Paulis satisfying ``X'_q C = C X_q``;
* ``z_images``: ``n`` signed output Paulis satisfying ``Z'_q C = C Z_q``;
* ``stabilizers``: ``m - n`` independent signed output Paulis satisfying
  ``S C = C``.

These data specify the isometry up to global phase. The output register size
is ``n_outputs`` (also ``n_qubits``), and ``n_inputs`` counts the logical inputs.
With no extra outputs this is the usual unitary Clifford tableau. A state
preparation has no logical images and is specified by its output stabilizers.
The empty tableau represents the one-dimensional identity.

Construction checks dimensions, canonical logical commutation relations,
commutation with stabilizers and stabilizer independence. Invalid Clifford
maps raise ``CliffordMapError``, a subclass of ``ValueError``.

Basic circuit-to-tableau conversion
-----------------------------------

.. code-block:: python

    import pyzx as zx
    from pyzx.circuit.gates import InitAncilla

    circuit = zx.Circuit(1)
    circuit.add_gate(InitAncilla(1, state="0"))
    circuit.add_gate("CNOT", 0, 1)
    tableau = zx.CliffordMap.from_circuit(circuit)
    print(tuple(map(str, tableau.x_images)))     # ('+XX',)
    print(tuple(map(str, tableau.z_images)))     # ('+ZI',)
    print(tuple(map(str, tableau.stabilizers)))  # ('+ZZ',)

``from_circuit`` propagates the signed generators directly through the gate
list. It performs no graph conversion, flow finding or dense matrix evaluation.
It supports ``HAD``, ``CNOT``/``CX``, ``CZ``, ``XCX``, ``SWAP``, ``Y`` and
``XPhase``/``ZPhase`` angles that are exact multiples of pi/2. This includes
``NOT``, ``Z``, ``S``/``Sdg`` and ``SX``/``SXdg``. Numeric phases are checked
exactly without rounding floats to a nearby Clifford angle. Other gates,
symbolic or non-Clifford phases, classical wires, measurements, resets and
postselection are rejected. It does not recognize a globally Clifford circuit
whose individual gates fall outside this supported set.

Leading ``InitAncilla`` gates may prepare ``+``, ``-``, ``0`` or ``1`` on
consecutive wires after the input register. All preparations must precede all
other gates. Use these explicit preparations instead of the
``Circuit.initialize_qubits`` flags for this conversion.

Reference tableau-to-circuit synthesis
--------------------------------------

.. code-block:: python

    synthesized = tableau.to_circuit()

This method implements a basic reference algorithm, not an optimized
Clifford compiler:

1. Complete an isometry to a unitary tableau. The output stabilizers are used
   as the extra logical X images; missing Z partners are found by binary
   elimination and adjusted to commute. The completion outside the isometry's
   image is arbitrary.
2. Reduce that unitary tableau to the identity using H, S, X/Z, CNOT and SWAP
   conjugations. Reverse and adjoint the reduction gates to obtain a circuit.
3. For an isometry, prepend ``InitAncilla`` preparations of ket plus on wires
   ``n_inputs`` through ``n_outputs - 1``.

The returned ``Circuit.qubits`` counts the original open inputs, while the
initialization gates introduce the extra output wires. The synthesized
preparations need not match those in a source circuit. The entire isometry is
preserved up to a nonzero global scalar under PyZX's graph conventions.
Calling ``to_basic_gates`` additionally expands SWAP gates into three CNOTs.
Circuit export uses the existing circuit exporters and their supported gates;
there is no dedicated tableau export format in this contribution.

Small-instance matrix verification
----------------------------------

``CliffordMap.to_matrix(max_qubits=8)`` constructs a normalized common +1
eigenvector of the logical Z images and output stabilizers, then obtains the
remaining columns using the logical X images. This evaluates the tableau
directly without gate synthesis. The matrix has shape
``(2**n_outputs, 2**n_inputs)`` and satisfies ``C.conj().T @ C = I``.
Wire zero is the most significant tensor factor. Dense evaluation is
exponential and is intended only for small examples; construction and
reference circuit synthesis do not use it.
