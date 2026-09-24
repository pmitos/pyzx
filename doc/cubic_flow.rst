Cubic flow finding for XY, X and Y
=================================

``pyzx.gflow.gflow`` uses incremental column elimination by default
(``method="cubic"``); ``method="legacy"`` retains the previous PyZX finder.
The incremental finder uses packed Python integers without new dependencies.
It returns focused corrections even when ``focus=False``; correction choices
and layer numbers can differ from legacy.

Why XY/X/Y permits this method
-----------------------------

Let R=V\\O and S=V\\I. The flow-demand matrix M has rows R and columns S.
XY and X rows contain adjacency; Y rows additionally contain a diagonal one
when their vertex is not an input. The order-demand matrix N selects the
correction coordinate of each non-input XY vertex and is zero on Pauli rows.

Thus N consists of zero rows and an identity subblock. Its order constraints
simply exclude individual unprocessed XY correction coordinates. This is
stronger than merely being sparse: general XZ/YZ rows can impose parity
constraints allowing cancellations, which this column-availability rule
does not handle. The M/N formulation is from Mitosek and Backens,
https://arxiv.org/abs/2410.23439.

Incremental construction
------------------------

Let T be the measured vertices already placed in later layers. Available
correctors are A=(O union X union Y union T) minus I. For each remaining u,
solve M[R,A] c=e_u. All rows R remain present, including previously solved
vertices, so focusing constraints are retained.

Maintain one elimination basis of the available M columns. Each basis entry
stores b=Mq, where q records its combination of original correction columns.
For every unsolved u, maintain a partial correction c_u and residual r_u with
M c_u + r_u=e_u.

Each newly available column is reduced against the existing basis. A dependent
column adds no new freedom. If it introduces a pivot, eliminate that pivot
from every affected unsolved residual, applying the same XOR to its correction
coordinates. Collect all zero-residual targets as one batch before unlocking
their XY columns. No progress with unsolved targets remaining means failure.

There is no global C0, kernel K, parameter matrix P, or identity block for
order-constraint removal. Each column is inserted at most once and each pivot
updates each unsolved target at most once. These are O(n^2) packed-vector
operations on O(n)-bit integers: O(n^3) bit work and O(n^2) matrix bits.
Returned correction sets have O(n^2) entries in the worst case.

Grounds and numbering
--------------------

Grounds are initially available correctors, receive the output layer and have
no correction-map entry. With ``focus=False``, their demand rows are omitted.
With ``focus=True``, those rows remain homogeneous constraints, but their
unit right-hand sides are not correction targets. The incremental method
handles both conventions directly, without a legacy fallback.

Normal numbering runs from earlier measurements to later ones; reverse mode
swaps input/output roles and reverses the layer-numbering convention.

Balanced-case option and reference backend
-----------------------------------------

When |I|=|O| on a ground-free graph, M is square. Mbqcflow's approach is then
particularly simple: compute C=M^-1, reject singular M, and check NC for cycles.
Since N selects rows, NC does not require general multiplication for XY/X/Y.
In fact, let T be the measured, non-input XY vertices. Every other row of NC
is zero, so a cycle exists exactly when the principal submatrix C[T,T] has a
cycle. The square reference path checks only this restriction for cycles and
then assigns layers to the other vertices from their edges into T.

The previous M/N port is retained privately as ``_gflow_matrix``, with its
matrix-algebra tests. Its rectangular path constructs C0 and K, maintains
[NK | NC0 | I] to obtain P, and returns C=C0+KP. Focused non-output grounds
fall back to legacy in this reference backend only.

The public finder does not dispatch balanced cases to this backend: its Python
inverse-and-DAG implementation was slower than incremental on the local suite.
An accelerated square inverse could support a future hybrid, but must be
benchmarked end to end, including correction-set conversion. NumPy is already
a PyZX dependency; Numba is not. No new accelerated square backend is currently
enabled.

Verification
------------

Tests check all 4,233 tiny graph/input/output/XY-X-Y labelling combinations
through three vertices against independent correction-set and total-order
enumeration. Seeded graph/API-mode checks compare incremental, the private
M/N backend and legacy, and validate successful witnesses independently.

Separate dense square/rectangular fixtures and multilayer cases cover both
implementations. Algebra tests retain MC0=I, MK=0, nonzero kernel adjustments
needed to repair cycles, and parameter/order enumeration. Dispatch tests
ensure the public default remains incremental for both boundary shapes.

Other cases cover rank failure, cycles, zero/dependent columns, phase
periodicity, input exclusion, disconnected components, grounds, boundary-only
diagrams, reverse numbering, X spiders, sparse IDs, both graph backends,
repeatability and absence of graph mutation. Pauli-web integration checks
validate the actual finder witness in both circuit directions.

The task's benchmark reports preserve the comparisons of commits 942dbfc
(incremental) and 5183e0e (M/N port) with mbqcflow auto. Those scripts load both
historical snapshots explicitly, so changing the default does not silently
change the benchmarked algorithms. Proper LUH benchmarking remains planned.
