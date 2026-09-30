Incremental flow finding for XY, X and Y
========================================

Use ``pyzx.gflow.gflow`` for ordinary XY gflow and ``pyzx.gflow.pauli_flow``
for Pauli flow with XY, X and Y measurements. The latter infers X and Y from
spider phases; other measurement types are not supported. The old
``gflow(..., pauli=True)`` call delegates to ``pauli_flow`` for compatibility.

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
the incremental and legacy finders on seeded graphs, independently check
returned correction witnesses, and cover rank and order failure, grounds,
wide packed vectors, many layers, both graph backends and Pauli-web callers.
