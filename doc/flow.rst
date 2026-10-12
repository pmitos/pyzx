Flow finding
============

``pyzx.flow.gflow`` finds gflow by interpreting all spiders in a graph-like diagram
as :math:`XY` measurements. ``pyzx.flow.pauli_flow`` finds Pauli flow by interpreting
Pauli phases as :math:`X` or :math:`Y` measurements and other phases as :math:`XY`.
Both finders default to ``method="cubic"``, which runs in :math:`O(n^3)` time for
:math:`n` vertices and returns focused corrections.

``layers_only=True`` returns the partial-order layers without correction sets. It
can save time when only flow existence or the layers are needed:

.. code-block:: python

    from pyzx.flow import pauli_flow

    layers = pauli_flow(g, layers_only=True)
    has_flow = layers is not None

The ``focus`` option belongs to ``method="legacy"``. That finder supports both
focused and unfocused corrections. For the cubic finder, use
``include_ground_constraints=True`` to retain homogeneous constraints on ground
vertices; these constraints are omitted by default. Ground vertices are processed
with outputs and have no correction sets of their own in either mode.

See :mod:`pyzx.flow` for the full API reference. Existing imports from
``pyzx.gflow`` remain supported.

Cubic algorithm
---------------

The following overview assumes a diagram without ground generators.

The main idea follows the algebraic formulation by Mitosek and Backens, DOI
10.1088/1751-8121/ae2999. However, the implementation uses additional optimisation by
exploiting the special structure of the order-demand matrix.

Let :math:`V` be the vertices of the open graph, with input set :math:`I` and output
set :math:`O`. Set :math:`R=V\setminus O` and :math:`S=V\setminus I`. All matrices
are over :math:`\mathbb{F}_2`. The flow-demand matrix :math:`M` has rows :math:`R`
and columns :math:`S`. :math:`XY` and :math:`X` rows of :math:`M`
contain adjacency; :math:`Y` rows of :math:`M` additionally contain a diagonal one when
their vertex is not an input. The order-demand matrix :math:`N` also has rows :math:`R`
and columns :math:`S`. With only :math:`XY/X/Y` measurements, the only non-zero entries
of :math:`N` are at the intersection of internal :math:`XY` measurements. The
correction matrix :math:`C` has rows :math:`S` and columns :math:`R`; its column for
:math:`u` encodes the correction set of :math:`u`. The formulation states that flow
exists if :math:`MC=\mathrm{Id}_R`, the identity matrix on :math:`R`, and :math:`NC`
forms the adjacency matrix of a directed acyclic graph (DAG).

The construction finds :math:`C` layer by layer, starting from the outputs.

:math:`N` is so simple that we don't need to construct it. Furthermore, instead of the
construction in DOI 10.1088/1751-8121/ae2999, which first constructs a right inverse
:math:`C` of :math:`M` and then adjusts it to make :math:`NC` a DAG, we can restrict the
construction of :math:`C` to ensure :math:`NC` is a DAG during construction. This
approach is fast only when there are no :math:`XZ/YZ/Z` measurements, which applies to
graph-like diagrams. As a result, the algorithm here is slightly faster than an
implementation that would closely follow Mitosek--Backens, though still :math:`O(n^3)`.

Furthermore, the implementation uses packed integer representation. Instead of holding
matrix :math:`M` as a two-dimensional array, we hold an array of integers. Each integer,
when written in binary, corresponds to the column of the matrix.

Layer-by-layer construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`X` and :math:`Y` be the sets of vertices measured in those bases, and
:math:`T` the measured vertices already placed in later layers. Available
correctors are :math:`A=(O\cup X\cup Y\cup T)\setminus I`. For each unsolved :math:`u`,
solve :math:`M[R,A]c=e_u`, where :math:`e_u` is the unit vector for row :math:`u`.
Maintain one elimination basis of the available :math:`M` columns. Each basis entry
stores :math:`b=Mq`, where :math:`q` records a combination of
original correction columns. For every unsolved :math:`u`, maintain a partial correction
:math:`c_u` and residual :math:`r_u` with :math:`Mc_u+r_u=e_u` over
:math:`\mathbb{F}_2`.

When moving to earlier layers, the columns considered in :math:`M[R,A]c` grow. We add a
column only when it becomes relevant after adjusting it against the existing basis. If
the column introduces a pivot in the reduced matrix, eliminate that pivot from every
affected unsolved residual, applying the same XOR to its correction coordinates.
Otherwise, the core methodology resembles previous flow-finding algorithms.

With ``layers_only=True``, the implementation still detects which vertices can be solved
as usual. This involves checking that the relevant linear system is consistent. Then,
the implementation skips tracking correction coordinates and decoding the actual
correction set, saving time.
