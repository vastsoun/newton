.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Kamino
======

:class:`~newton.solvers.SolverKamino` simulates constrained rigid multi-body
systems in maximal coordinates. It is designed for mechanical assemblies with
kinematic loops, under- or overactuation, joint limits, hard frictional
contacts, and restitutive impacts.

Unlike the other maximal-coordinate solvers, Kamino focuses on constrained
rigid mechanical assemblies rather than particle or deformable simulation.
Kamino is currently in BETA 1, and Newton users are discouraged from depending
on it. Evaluate it only when kinematic loops and hard contact constraints are
primary requirements and an experimental solver is acceptable.

.. experimental::

   :class:`~newton.solvers.SolverKamino` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

See the :class:`~newton.solvers.SolverKamino` API reference for construction
and configuration details. Runnable workflows are available in the
`Kamino examples <https://github.com/newton-physics/newton/tree/main/newton/examples/kamino>`_.

Choosing a dynamics solver
--------------------------

Kamino provides two forward-dynamics backends:

* ``"padmm"`` (default): proximal ADMM, dense Jacobians/dynamics, and the Euler
  integrator. It is the slower, more robust option because it solves equality
  and inequality constraints together.
* ``"dvi"`` (opt-in): projected dual iterations, sparse Jacobians, dense dynamics
  with the RCM-reordered blocked LLT solver, and the Euler integrator. It is
  generally faster, but approximates the coupled problem by alternating between
  a direct solve for equality constraints and projected iterations for
  inequality constraints. As a rule of thumb, DVI solves inequality constraints
  less accurately than PADMM, particularly as the number of active inequalities
  grows. Dual preconditioning is not supported.

Select the backend when constructing the configuration so dependent defaults
initialize consistently:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   solver = newton.solvers.SolverKamino(model, config=config)

DVI is best suited to performance-sensitive rigid mechanisms with relatively
few active contacts; PADMM remains the safer and more broadly validated choice.
Set ``sparse_jacobian=False`` for fully dense DVI, or set
``sparse_dynamics=True`` to use sparse dynamics with the Conjugate Residual
solver. With
``collect_solver_info=True``, DVI stores terminal residual status that should
not be interpreted as PADMM ADMM residuals.

For large bilateral systems, opt into RCM-reordered factorization explicitly:

.. code-block:: python

   config.dvi.bilateral_solver_type = "LLTBRCM"
   config.dvi.bilateral_solver_kwargs = {
       "block_size": 32,
       "reuse_permutation": True,
       "parallel_factorization": True,
   }

The cached permutation remains mathematically valid when matrix values or
sparsity change and is recomputed automatically if the active dimension
changes. Keep the default ``"LLTB"`` solver for small systems.

Contact-buffer capacity
-----------------------

All Kamino contact-buffer allocations funnel through a single resolver,
:meth:`~newton._src.solvers.kamino._src.geometry.capacity.ContactCapacity.resolve_from`,
which returns an immutable
:class:`~newton._src.solvers.kamino._src.geometry.capacity.ContactCapacity`
describing literal per-world buffers whose sum is the model total. Every
construction path (standalone
:class:`~newton._src.solvers.kamino._src.geometry.CollisionDetector`,
:class:`~newton.solvers.SolverKamino` with internal PADMM or DVI collision
detection, and :class:`~newton.solvers.SolverKamino` in external
Newton-contacts mode) consumes the same resolver.

Which policy is applied depends on the coupling and the dynamics backend:

* **Full internal (PADMM, default)** — Per-world capacity equals the
  geometry-pair estimate produced by the Kamino model builder
  (``model.geoms.world_minimum_contacts``, computed from
  :func:`~newton._src.solvers.kamino._src.core.geometry.max_contacts_for_shape_pair`).
  The narrow ``broadphase="explicit"`` / group-based fallback only applies
  when a standalone ``ModelKamino`` lacks pair metadata; Newton-derived
  models always populate it.
* **Bounded internal (DVI)** — Per-world capacity is
  ``min(geometry, Newton bounded heuristic)``. The heuristic reuses the
  same constants as
  :func:`newton._src.sim.collide._estimate_rigid_contact_max` and preserves
  global-geometry accounting per world. Worlds whose geometry cannot
  produce contacts receive a literal zero budget. This intentional
  divergence from the full-internal policy keeps DVI's dense storage
  bounded for scenes whose geometry-pair count grows quadratically.
* **External Newton (``use_collision_detector=False``)** — The resolver
  honors :attr:`~newton._src.sim.model.Model.rigid_contact_max` exactly
  when nonzero. Otherwise it defers to
  :func:`newton._src.sim.collide._estimate_rigid_contact_max`. The
  resulting model total is distributed across worlds using geometry-pair
  weights via largest-remainder rounding, so the per-world sum equals the
  model total exactly (previous releases rounded up to a multiple of
  ``world_count``, e.g. ``1000 -> 1002``).

The following knobs (all on
:class:`~newton._src.solvers.kamino.config.CollisionDetectorConfig`) apply
in order for **internal** policies only:

1. ``max_contacts_per_world`` — Uniform per-world override. Applied first
   and bypasses every other input, including ``max_contacts``. Intended
   for tests and memory-budgeted runs.
2. Policy sizing — Full geometry estimate (PADMM) or per-world
   ``min(geometry, bounded)`` (DVI).
3. ``max_contacts`` — Model-wide cap. When set and exceeded, per-world
   budgets are scaled down proportionally by largest-remainder rounding so
   that the total exactly matches ``max_contacts``.
4. ``max_contacts_per_pair`` — Narrow-phase cap on the number of contacts
   any single geom-pair can generate. This flows through
   :func:`~newton._src.solvers.kamino._src.core.geometry.max_contacts_for_shape_pair`
   and reduces the geometry estimate that feeds the resolver, so lowering
   it also lowers the resulting per-world buffer.

For external Newton contacts, ``model.rigid_contact_max`` is the single
input. The detector-config sizing fields above are internal-only.

Buffer overflow is a runtime concern, not a construction-time one. When
active contacts exceed the per-world capacity in a single step, the
collision detector drops the overflowing contacts and emits a single
warning per detector (see `PR #3791
<https://github.com/newton-physics/newton/pull/3791>`_). If you observe
this warning, raise ``max_contacts_per_world`` or
:attr:`~newton._src.sim.model.Model.rigid_contact_max`, not
``max_contacts_per_pair``.

Migration guidance:

* Kamino previously derived ``model.geoms.model_minimum_contacts`` from
  ``model.rigid_contact_max`` when it was already set. This construction
  order dependency is gone: conversion always uses the shape-pair
  geometry, and the resolver treats ``rigid_contact_max`` as an input
  only for external Newton mode. Set ``rigid_contact_max`` when you own
  the Newton-side buffer; use ``max_contacts_per_world`` /
  ``max_contacts`` for internal-mode sizing.
* DVI previously flattened multi-world budgets to a single maximum. It
  now returns heterogeneous per-world budgets that follow geometry
  (worlds with more shapes get bigger buffers). Constraint / Jacobian
  sizing derives from the resulting ``ContactsKamino`` container, so no
  downstream config changes are required.
