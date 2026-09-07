Cubic flow finding for XY, X and Y
=================================

``pyzx.gflow.gflow`` accepts ``method="cubic"`` (the default) and
``method="legacy"``. The cubic finder ports mbqcflow's right-inverse/kernel
algorithm using Python integers as packed binary rows, without new dependencies.
It supports XY, X and Y measurements, the return shape and reverse-numbering
convention of the previous finder. Corrections are focused even when
``focus=False``; correction choices and layers can differ from legacy.

Algebra
-------

Rows of M are non-outputs; columns are non-inputs. XY and X rows contain
adjacency; Y rows additionally contain a diagonal one when their vertex is
not an input. The order-demand matrix N selects the correction coordinate of
each non-input XY vertex and is zero on Pauli rows. Thus multiplying by N
only requires selecting rows, not general matrix multiplication. These are
the XY/X/Y cases of Mitosek and Backens, https://arxiv.org/abs/2410.23439.

The task is to find C with MC=I and NC a DAG matrix. Entry (v,u) of NC records
the required dependency from u to v, so diagonal entries are forbidden too.
The implementation follows the square/general split in mbqcflow's
``find_dag_right_inverse_gf2_with_layers``.

Equal input and output counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

M is square. Reduce [M | I] to find its inverse, or reject singular M.
The only possible C is M^-1. Check NC for cycles and obtain sink-first layers.
There is no kernel construction, parameter matrix P, or incremental system
to solve. This bypass is explicitly enforced by the unit tests.

More outputs than inputs
~~~~~~~~~~~~~~~~~~~~~~~~

Let r be the row count, c the column count and k=c-r. Reduction of [M | I]
gives a right inverse C0 (with free variables zero). If M does not have full
row rank, reject. The same reduction supplies a c-by-k kernel basis K.
Every right inverse is C=C0+KP, with P of shape k-by-r.

Set A=NK and B=NC0. Port mbqcflow's ``_find_p_matrix_gf2_with_layers`` using
the maintained system [A | B | I]:

* Initially put the A block into row echelon form, applying the same operations
  to B and I.
* A target u is solvable precisely when column u of the transformed B is zero
  on every zero-coefficient row. Solve all currently solvable targets together
  as one batch, using back substitution to obtain their columns of P.
* Remove the original constraints belonging to that batch. The identity block
  records which transformed rows use each original constraint. Choose the last
  such row, XOR it out of the others, then XOR the original invariant row out
  of that replacement row. Reduce and reinsert only the changed row to restore
  echelon order. This is mbqcflow's identity-block update, not a fresh solve.
* Repeat, rejecting if unsolved targets remain but none is solvable.

Finally form C=C0+KP and check NC for cycles, as mbqcflow does. Sink-first
layers of this product provide the public layer numbering. More inputs than
outputs cannot give a right inverse and are rejected before elimination.

Grounds
-------

Grounds receive the output layer and have no correction-map entry. With
``focus=False``, they are treated as extra outputs: omit their M and N rows
and use the same algorithm. Matrix dimensions, rather than boundary counts
alone, determine which path is used.

With ``focus=True``, a non-output ground imposes a homogeneous constraint but
has no unit right-hand side of its own. This is not the MC=I problem. That
case explicitly falls back to the existing legacy finder to preserve PyZX's
convention; it does not have the cubic complexity guarantee. If all grounds
are already outputs, no fallback is needed.

Complexity
----------

The port retains O(n^3) bit work and O(n^2) packed matrix bits, where n is the
number of internal vertices. Reduction, kernel construction and KP formation
each fit this bound. There are at most r constraint removals, each requiring
O(r) packed row operations. Back substitution uses at most k packed parity
evaluations per target, with each target solved only once; it is not a complete
elimination at every layer. Packed rows have O(n) bits throughout. The square
case omits all kernel and maintained-system work. Returned Python correction
sets have O(n^2) entries in the worst case.

Verification
------------

The flow tests check all 4,233 graph/input/output/XY-X-Y labelling combinations
through three vertices against correction-set and total-order enumeration,
independently of focusing or elimination. There are also 1,280 seeded graph/API
mode comparisons with legacy and independent parity/order checks of witnesses.

Algebra tests check MC0=I and MK=0, enumerate parameter columns and total orders
for small A/B systems, verify that square inputs bypass kernel adjustment, and
exercise a rectangular example where nonzero P is necessary to remove a cycle.
Separate dense square and rectangular cases cross 64-bit boundaries; 67
disjoint chains exercise a wide kernel with repeated constraint removals.

Further cases cover rank failure, diagonal and longer cycles, zero/dependent
columns, phase periodicity, input exclusion, disconnected components, grounds,
boundary-only diagrams, reverse numbering, X spiders, sparse vertex IDs, both
simple and multigraph backends, repeatability and absence of input mutation.
An integration test validates the actual witness used by ``compute_pauli_webs``
in both directions on a circuit with Clifford and non-Clifford gates.

The local verification scripts additionally compare matrix and graph instances
directly with mbqcflow and exercise Pauli-web construction on reduced circuits.
Quick finder-only timings are recorded separately; representative benchmarking
is deferred to the LUH cluster.
