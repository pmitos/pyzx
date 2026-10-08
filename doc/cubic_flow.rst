Incremental flow finding for XY, X and Y
========================================

Use ``pyzx.flow.gflow`` for ordinary XY gflow and ``pyzx.flow.pauli_flow``
for Pauli flow with XY, X and Y measurements. The latter infers X and Y from
spider phases; other measurement types are not supported. The old
``gflow(..., pauli=True)`` call delegates to ``pauli_flow`` for compatibility.
Both entry points and the auxiliary finder live in ``pyzx/flow.py``.
The historical ``pyzx.gflow`` module re-exports the public entry points so
existing imports continue to work.

Each entry point prepares explicit measurement assignments before calling
the shared auxiliary ``_find_incremental_flow``. ``gflow`` assigns XY to
every spider regardless of phase; ``pauli_flow`` infers XY/X/Y assignments.
The auxiliary finder uses those assignments without reading phases. Both
entry points also assign grounded spiders, whose rows may be retained as
homogeneous constraints by ``focus=True``.

Both entry points use incremental column elimination by default. The
``method="cubic"`` default and ``method="incremental"`` name select the same
finder for every boundary shape, including balanced graphs. The previous
finder remains available as ``method="legacy"``. The incremental finder uses
packed Python integers without new dependencies. It returns focused
corrections even when ``focus=False``; correction choices and layer numbers
can differ from legacy.

Use ``layers_only=True`` when correction sets are unnecessary. This keyword
works with ``cubic`` and ``incremental`` and returns a layer
dictionary on success, or ``None`` on failure. Without it, the return value
remains ``(layers, corrections)`` or ``None``. ``legacy`` rejects this option.

For example::

    from pyzx.flow import pauli_flow

    layers = pauli_flow(g, method="incremental", layers_only=True)
    has_flow = layers is not None

Check against ``None`` for existence: an empty graph has flow and returns
an empty layer dictionary, which is false in a Boolean context. Ordinary
gflow supports the same keyword, as does the ``gflow(..., pauli=True)`` alias.

Why XY/X/Y permits incremental elimination
-------------------------------------------

Let R=V\\O and S=V\\I. The flow-demand matrix M has rows R and columns S.
XY and X rows contain adjacency; Y rows additionally contain a diagonal one
when their vertex is not an input. The order-demand matrix N selects the
correction coordinate of each non-input XY vertex and is zero on Pauli rows.

Thus N consists of zero rows and an identity subblock. Its order constraints
exclude individual unprocessed XY correction coordinates. General XZ/YZ rows
can impose parity constraints allowing cancellations, so this availability
rule does not handle them. The M/N formulation is from Mitosek and Backens,
https://arxiv.org/abs/2410.23439.

Incremental construction
------------------------

Let T be the measured vertices already placed in later layers. Available
correctors are A=(O union X union Y union T) minus I. For each remaining u,
solve M[R,A] c=e_u. All demand rows remain present, including rows for
previously solved vertices, so focusing constraints are retained.

Maintain one elimination basis of the available M columns. Each basis entry
stores b=Mq, where q records its combination of original correction columns.
For every unsolved u, maintain a partial correction c_u and residual r_u with
M c_u + r_u=e_u over GF(2).

Reduce each newly available column against the existing basis. A dependent
column adds no new freedom. If it introduces a pivot, eliminate that pivot
from every affected unsolved residual, applying the same XOR to its correction
coordinates. Collect all zero-residual targets as one batch before unlocking
their XY columns for the next layer. No progress with unsolved targets
remaining means failure.

Each column is inserted at most once and each pivot updates each unsolved
target at most once. These are O(n^2) packed-vector operations on O(n)-bit
integers: O(n^3) bit work and O(n^2) matrix bits. Returned correction sets
have O(n^2) entries in the worst case.

Layers without corrections
--------------------------

Solvability depends only on whether each unit right-hand side reduces to zero
against the basis of permitted matrix columns. The correction coordinates
carried along during elimination provide witnesses but do not affect pivot
selection, residuals, or which vertices are solved together.

With ``layers_only=True``, skip creation and XOR updates of packed column
combinations and target solutions, and skip correction-set decoding. Basis
entries keep a zero placeholder for coordinates; the solution list is empty.
All demand rows and residual operations remain present, including homogeneous
ground constraints. The returned layers and existence decision are identical
to incremental flow finding with corrections enabled.

The O(n^3) bit-work and O(n^2) storage bounds remain unchanged. This removes
witness bookkeeping and decoding; its runtime benefit needs benchmarking.
Layers-only results do not include correction witnesses for external checking.

Grounds and numbering
--------------------

Grounds are initially available correctors, receive the output layer and
have no correction-map entry. With ``focus=False``, their demand rows are
omitted. With ``focus=True``, those rows remain homogeneous constraints, but
their unit right-hand sides are not correction targets.

Normal numbering runs from earlier measurements to later ones; reverse mode
swaps input/output roles and reverses the layer-numbering convention.

Verification
------------

Tests enumerate all 4,233 graph/input/output/XY-X-Y combinations through
three vertices against the Pauli-flow axioms directly. Other tests compare
the incremental and legacy finders on seeded graphs,
independently check returned correction witnesses, and cover rank and order
failure, grounds, wide packed vectors, many layers, both graph backends and
Pauli-web callers.
Layers-only results are checked against the full finders' exact layers and
existence decisions across the exhaustive cases, seeded modes, grounds,
explicit assignments, wide vectors, many layers and the compatibility alias.

Reproducing a performance example
--------------------------------

The fixed ``XY_X_Y/n00200-r00`` example from UnlabelledOpenGraphs is included
as ``tests/data/flow/xy_x_y_n200_r00.json``. It has 200 spiders and 28 explicit
input/output boundary vertices. Ordinary tests check its incremental witness,
layers-only result, and all-XY failure without running the slow legacy finder
or asserting a runtime bound.

From a source checkout, run the optional comparison::

    python -m tests.benchmark_flow --repeats 3 --output flow-benchmark.json

Both methods use ``pauli_flow(..., focus=True)`` with full correction results,
the same interpreter and graph, and alternating call order. Only finder calls
are timed. Loading, graph construction, warm-up, mutation checks and independent
witness validation are excluded. The JSON report records every repetition,
wall and CPU medians, source/fixture hashes, and the runtime environment.
Timings illustrate this fixed instance; they do not establish performance on
arbitrary graphs. Fixture provenance is documented in
``tests/data/flow/README.md``.
