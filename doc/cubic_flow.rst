Cubic flow finding for XY, X and Y
=================================

``pyzx.gflow.gflow`` accepts ``method="cubic"`` (the default) and
``method="legacy"``. The cubic finder uses Python integers as packed binary
vectors and adds no dependency. It returns focused corrections even if
``focus=False``; correction choices and layer numbers can differ from legacy.
The return shape and reverse-numbering convention are preserved.

Algebra
-------

For a graph without grounds, rows of M are non-outputs and columns are
non-inputs. XY and X rows contain adjacency; Y rows additionally contain a
diagonal one when their vertex is not an input. The order-demand matrix N
selects the correction coordinate of each non-input XY vertex and is zero
on Pauli rows. This is the XY/X/Y specialization of the formulation in
Mitosek and Backens, https://arxiv.org/abs/2410.23439.

At each layer, eligible correction columns are outputs, Pauli vertices and
vertices solved in earlier layers, excluding inputs. Solving M c = e_u using
only those columns both enforces focused parity and prevents dependencies
on unprocessed XY vertices. Collect all currently solvable vertices before
unlocking their columns, so the resulting order is strict.

Rather than construct a general right inverse and kernel basis as in the
mbqcflow implementation, this specialization incrementally maintains a basis
of the eligible columns. A basis element stores its transformed M column and
the combination of original columns representing it. New elements are reduced
against existing elements in insertion order. Thus a new element is zero at
every earlier pivot (pivot indices need not be numerically increasing).

For every unsolved u, maintain residual r_u and correction c_u with invariant
M c_u XOR r_u = e_u. Each new pivot eliminates that coordinate from every
residual, applying the same XOR to its correction. Residual zero means solved.
A nonzero residual has no existing pivot coordinate and cannot lie in the
span of the basis; no solution is lost. An omitted dependent column adds
nothing to the span. If progress stops, no remaining vertex can be the next
sink of a focused flow. Existence of a focused flow is sufficient for the
supported Pauli-flow problem.

Grounds follow the previous API convention: they are available from the
start, receive layer zero before reversing the numbering, and have no
correction map entry. With focus=True their rows remain as homogeneous
constraints; with focus=False they are omitted, like output rows. This is a
PyZX convention, distinct from the ground-free open-graph theorem above.

Complexity
----------

There are at most n inserted columns and n independent pivots. Each column
is reduced at most once against each pivot. Each pivot updates at most n
residuals. These are O(n^2) vector operations on O(n)-bit integers, hence
O(n^3) bit operations (or O(n^3/w) word work for the packed XORs). Layer scans
and correction decoding also fit O(n^3). The stored column vectors, basis,
residuals and witnesses use O(n^2) bits; returned Python sets have O(n^2)
entries in the worst case.

Local verification, 7 September 2026
-----------------------------------

* 527 unit tests run, 13 skipped, no failures; mypy checks 150 source files.
* 1,280 seeded graph/mode combinations agree with legacy on existence;
  returned cubic witnesses are independently checked for parity and order.
* 2,000 gflow and 2,000 XY/X/Y Pauli-flow instances agree with mbqcflow.
* 48 direct Pauli-web calls succeed: 12 reduced circuits, both directions,
  both finders. This smoke check does not assert identical Pauli webs.
* A single dense bipartite example with 320 spiders, 320 boundary vertices,
  and an invertible 160-by-160 demand matrix took median 0.005657 s (cubic)
  versus 5.592353 s (legacy): approximately 989-fold speedup. Both ran under
  Python 3.13.7 on the same graph; three warm finder-only runs per method,
  with construction excluded. This is a spot check, not a representative
  benchmark. Proper benchmarking is deferred to the LUH cluster.
