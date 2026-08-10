# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-(problem, solver) factories for the Kamino benchmark suite."""

from collections.abc import Callable
from typing import NamedTuple

import warp as wp

import newton
from newton import solvers
from newton._src.core import Axis
from newton._src.sim import ModelBuilder
from newton._src.solvers.kamino._src.metrics import SolutionMetricsLogger
from newton._src.solvers.kamino.accuracy_benchmark.setup import SolverSetup
from newton.tests.utils import basics

###
# Module interface
###

__all__ = [
    "ProblemRun",
    "apply_external_force_to_box",
    "build_box_on_plane_run",
    "build_fourbar_run",
    "make_external_force_cb_box_on_plane",
    "make_setup_box_on_plane_kamino",
    "make_setup_box_on_plane_mujoco",
    "make_setup_box_on_plane_xpbd",
    "make_setup_fourbar_kamino",
    "make_setup_fourbar_mujoco",
    "make_setup_fourbar_xpbd",
]


class ProblemRun(NamedTuple):
    """Everything :class:`SetupRunner` needs to drive one benchmark problem.

    Attributes:
        setups: Mapping of solver name to its :class:`SolverSetup`. ``"kamino"`` is
            always present; it's used as the leader inside the runner.
        force_cb: Optional ``force_cb(state, contacts)`` passed to
            :class:`SetupRunner`. ``None`` when the problem doesn't apply
            external body forces (e.g. fourbar, which is excited by joint actuators).
        camera: Optional ``(position, pitch, yaw)`` triple. ``None`` leaves the
            viewer's default camera in place.
    """

    setups: dict[str, SolverSetup]
    force_cb: Callable | None
    camera: tuple[wp.vec3, float, float] | None


###
# Box-on-plane: external-force kernel
###


@wp.kernel
def _apply_external_force_to_box(
    dt: wp.float32,
    friction: wp.float32,
    force_scale: wp.float32,
    force_start_time: wp.float32,
    force_stop_time: wp.float32,
    time: wp.array[wp.float32],
    gravity: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    num_active_contacts: wp.array[wp.int32],
    state_body_f: wp.array[wp.spatial_vectorf],
):
    nc = num_active_contacts[0]
    m = body_mass[0]
    g = wp.length(gravity[0])
    t = time[0]
    if t >= force_start_time and t < force_stop_time and nc > 0:
        f_ext = force_scale * m * g * friction
        state_body_f[0] = wp.spatial_vectorf(f_ext, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        state_body_f[0] = wp.spatial_vectorf()
    time[0] += dt


def apply_external_force_to_box(
    model,
    time: wp.array[wp.float32],
    contacts_in,
    state_out,
    dt: float,
    *,
    friction: float,
    force_scale: float,
    force_start_time: float,
    force_stop_time: float,
) -> None:
    """Launch the box-on-plane external-force kernel for one step.

    The kernel writes a planar friction-saturated push to ``state_out.body_f[0]`` while
    ``time`` is inside ``[force_start_time, force_stop_time)`` and at least one contact
    exists, and clears the body force otherwise. The kernel also advances ``time[0]``
    by ``dt`` so the launcher caller is *not* responsible for time-keeping.
    """
    wp.launch(
        kernel=_apply_external_force_to_box,
        dim=1,
        inputs=[
            dt,
            friction,
            force_scale,
            force_start_time,
            force_stop_time,
            time,
            model.gravity,
            model.body_mass,
            contacts_in.rigid_contact_count,
        ],
        outputs=[state_out.body_f],
        device=model.device,
    )


###
# Shared scene + reset helpers
###


def _build_problem_model(
    solver_type: type[solvers.SolverBase],
    scene_builder_fn: Callable,
    *,
    rigid_contact_max: int,
    scene_kwargs: dict | None = None,
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.1),
):
    """Build a (builder, model, q_base) triple for a benchmark problem.

    Registers the solver-type's custom attributes on the scene builder so the resulting
    model carries whatever extended state/contact buffers that solver expects, then
    runs ``scene_builder_fn(builder=..., **scene_kwargs)`` to populate the scene, wraps
    it in a world with the extended ``body_parent_f`` / ``force`` attributes, and writes
    ``base_position`` into the floating-base prefix of ``joint_q`` (when present).
    """
    # Coord-layout targets are required for correctness on floating-base models
    # (FREE joint has 7 coords / 6 dofs); with the legacy DoF-sized layout,
    # writes to `control.joint_target_q` would silently truncate. Setting the
    # flag here (before any builder is constructed) means every accuracy-
    # benchmark problem picks it up uniformly.
    newton.use_coord_layout_targets = True
    scene_kwargs = scene_kwargs or {}
    scene_builder = ModelBuilder(up_axis=Axis.Z)
    solver_type.register_custom_attributes(scene_builder)
    scene_builder_fn(builder=scene_builder, **scene_kwargs)

    builder = ModelBuilder(up_axis=Axis.Z)
    builder.request_state_attributes("body_parent_f")
    builder.request_contact_attributes("force")
    builder.add_world(scene_builder)

    q_base = wp.transformf(base_position, wp.quat_identity())
    builder.joint_q[:3] = [q_base.p[0], q_base.p[1], q_base.p[2]]
    if len(builder.joint_q) > 6:
        builder.joint_q[3:7] = [q_base.q[0], q_base.q[1], q_base.q[2], q_base.q[3]]

    model = builder.finalize(skip_validation_joints=True)
    model.rigid_contact_max = int(rigid_contact_max)
    return builder, model, q_base


def _make_kamino_reset_cb_with_base(solver: solvers.SolverKamino, q_base: wp.transformf) -> Callable:
    """Return a ``reset_cb(state_out)`` that resets Kamino state and sets the base pose.

    Uses :class:`SolverKamino.ResetConfig.FromBaseQ` to seed the floating-base
    pose from the same ``q_base`` written into ``builder.joint_q[:7]``.
    """
    base_q = wp.zeros(shape=(1,), dtype=wp.transformf)
    base_q.assign([q_base])
    config = solvers.SolverKamino.ResetConfig(
        base_pose=solvers.SolverKamino.ResetConfig.FromBaseQ(base_q),
    )

    def reset_cb(state_out):
        solver.reset(state=state_out, config=config)

    return reset_cb


def _make_fk_reset_cb(model, q_base: wp.transformf) -> Callable:
    """Return a ``reset_cb(state_out)`` that writes the base pose via forward kinematics.

    Used by solvers (MuJoCo, XPBD) that don't expose a dedicated reset op the way
    :class:`SolverKamino` does — write the floating-base pose into ``joint_q``, copy
    ``joint_qd`` from the model, then evaluate FK to populate ``body_q``/``body_qd``.
    """

    def reset_cb(state_out):
        joint_q_np = model.joint_q.numpy().copy()
        joint_q_np[:3] = [q_base.p[0], q_base.p[1], q_base.p[2]]
        if len(joint_q_np) > 6:
            joint_q_np[3:7] = [q_base.q[0], q_base.q[1], q_base.q[2], q_base.q[3]]
        state_out.joint_q.assign(joint_q_np)
        wp.copy(state_out.joint_qd, model.joint_qd)
        newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

    return reset_cb


def _make_setup(
    *,
    name: str,
    builder: ModelBuilder,
    model,
    solver: solvers.SolverBase,
    dt: float,
    rigid_contact_max: int,
    max_log_frames: int,
    reset_cb: Callable,
    friction: float,
) -> SolverSetup:
    """Construct a non-standalone :class:`SolverSetup` with a configured reset callback."""
    setup = SolverSetup(
        name=name,
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        standalone=False,
        kwargs_logger={
            "max_frames": max_log_frames,
            "mode": SolutionMetricsLogger.Mode.BOUNDED,
        },
    )
    setup.reset_cb = reset_cb
    # Stashed for downstream factories (e.g. make_external_force_cb_box_on_plane).
    setup._friction = friction
    return setup


###
# Box-on-plane + per-solver setups
###


def _box_on_plane_scene_kwargs(z_offset: float, friction: float, restitution: float) -> dict:
    return {
        "z_offset": z_offset,
        "friction": friction,
        "restitution": restitution,
        "use_custom_shape_cfg": True,
    }


def make_setup_box_on_plane_kamino(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    z_offset: float = 0.0,
    friction: float = 0.7,
    restitution: float = 0.0,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` for SolverKamino on box-on-plane."""
    builder, model, q_base = _build_problem_model(
        solvers.SolverKamino,
        basics.build_box_on_plane,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs=_box_on_plane_scene_kwargs(z_offset, friction, restitution),
        base_position=(0.0, 0.0, 0.1 + z_offset),
    )

    cfg = solvers.SolverKamino.Config()
    cfg.constraints.alpha = 0.0
    cfg.constraints.beta = 0.0
    cfg.constraints.gamma = 0.01
    cfg.constraints.delta = 1e-6
    cfg.dynamics.preconditioning = True
    cfg.padmm.use_acceleration = True
    cfg.padmm.warmstart_mode = "none"
    cfg.padmm.max_iterations = 1000
    cfg.padmm.rho_0 = 1.0
    cfg.compute_solution_metrics = True
    solver = solvers.SolverKamino(model=model, config=cfg)

    reset_cb = _make_kamino_reset_cb_with_base(solver, q_base)

    setup = _make_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=reset_cb,
        friction=friction,
    )
    # Second logger over the solver-internal SolutionMetrics. Surfaces in
    # SetupRunner.test_final as ``<problem>_kamino.pdf`` (internal vs front-end).
    setup.solver_logger = SolutionMetricsLogger(
        metrics=solver._solver_kamino.metrics,
        max_frames=max_log_frames,
        mode=SolutionMetricsLogger.Mode.ROLLING,
    )
    return setup


def make_setup_box_on_plane_xpbd(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    z_offset: float = 0.0,
    friction: float = 0.7,
    restitution: float = 0.0,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` for SolverXPBD on box-on-plane."""
    builder, model, q_base = _build_problem_model(
        solvers.SolverXPBD,
        basics.build_box_on_plane,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs=_box_on_plane_scene_kwargs(z_offset, friction, restitution),
        base_position=(0.0, 0.0, 0.1 + z_offset),
    )

    solver = solvers.SolverXPBD(
        model,
        iterations=2,
        soft_body_relaxation=0.9,
        soft_contact_relaxation=0.9,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.4,
        joint_linear_compliance=0.0,
        joint_angular_compliance=0.0,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=True,
        angular_damping=0.0,
        enable_restitution=False,
    )

    return _make_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model, q_base),
        friction=friction,
    )


def make_setup_box_on_plane_mujoco(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    z_offset: float = 0.0,
    friction: float = 0.7,
    restitution: float = 0.0,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` for SolverMuJoCo on box-on-plane."""
    builder, model, q_base = _build_problem_model(
        solvers.SolverMuJoCo,
        basics.build_box_on_plane,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs=_box_on_plane_scene_kwargs(z_offset, friction, restitution),
        base_position=(0.0, 0.0, 0.1 + z_offset),
    )

    solver = solvers.SolverMuJoCo(
        model,
        cone="elliptic",
        impratio=100,
        iterations=200,
        ls_iterations=100,
        nconmax=rigid_contact_max,
        njmax=100,
        use_mujoco_contacts=False,
    )

    return _make_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model, q_base),
        friction=friction,
    )


###
# External-force factory (problem-specific, runner-level hook)
###


def make_external_force_cb_box_on_plane(
    setup: SolverSetup,
    *,
    force_scale: float = 1.1,
    force_start_time: float = 0.0,
    force_stop_time: float = 4.0,
) -> Callable:
    """Return a ``force_cb(state, contacts)`` closure for use with :class:`SetupRunner`.

    The closure owns a one-element Warp ``time`` array which the kernel both reads
    (to gate the time window) and advances by ``setup.dt`` each call.
    """
    friction = getattr(setup, "_friction", 0.7)
    time_arr = wp.zeros(shape=(1,), dtype=wp.float32)

    def force_cb(state, contacts):
        apply_external_force_to_box(
            setup.model,
            time_arr,
            contacts,
            state,
            setup.dt,
            friction=friction,
            force_scale=force_scale,
            force_start_time=force_start_time,
            force_stop_time=force_stop_time,
        )

    return force_cb


###
# Four-bar linkage + per-solver setups
###


def _fourbar_scene_kwargs(limits: bool) -> dict:
    # ``floatingbase=True`` is load-bearing: it makes the model's ``joint_q``
    # prefix a 7-DoF transform so ``_make_fk_reset_cb`` (and the runner's pose
    # init in ``_build_problem_model``) can write the base pose into joint_q[:7].
    return {"floatingbase": True, "limits": limits, "ground": True}


def make_setup_fourbar_kamino(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    limits: bool = True,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` for SolverKamino on the box-fourbar linkage.

    Kamino config and the soft PD gains mirror
    :file:`newton/examples/kamino/example_kamino_basic_fourbar.py`.
    """
    builder, model, q_base = _build_problem_model(
        solvers.SolverKamino,
        basics.build_boxes_fourbar,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs=_fourbar_scene_kwargs(limits),
    )
    # Soft PD gains so the actuated revolutes don't snap-track and explode the
    # comparison; copied from example_kamino_basic_fourbar.py:71-72.
    model.joint_target_ke.fill_(1.0)
    model.joint_target_kd.fill_(0.001)

    cfg = solvers.SolverKamino.Config()
    cfg.dynamics.preconditioning = True
    cfg.padmm.use_acceleration = True
    cfg.padmm.warmstart_mode = "containers"
    cfg.padmm.contact_warmstart_method = "geom_pair_net_force"
    cfg.padmm.primal_tolerance = 1e-4
    cfg.padmm.dual_tolerance = 1e-4
    cfg.padmm.compl_tolerance = 1e-4
    cfg.padmm.max_iterations = 200
    cfg.padmm.rho_0 = 0.1
    cfg.compute_solution_metrics = True
    solver = solvers.SolverKamino(model=model, config=cfg)

    reset_cb = _make_kamino_reset_cb_with_base(solver, q_base)

    setup = _make_setup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=reset_cb,
        friction=0.0,
    )
    # Second logger over the solver-internal SolutionMetrics. Surfaces in
    # SetupRunner.test_final as ``<problem>_kamino.pdf`` (internal vs front-end).
    setup.solver_logger = SolutionMetricsLogger(
        metrics=solver._solver_kamino.metrics,
        max_frames=max_log_frames,
        mode=SolutionMetricsLogger.Mode.ROLLING,
    )
    return setup


def make_setup_fourbar_xpbd(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    limits: bool = True,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` for SolverXPBD on the box-fourbar linkage."""
    builder, model, q_base = _build_problem_model(
        solvers.SolverXPBD,
        basics.build_boxes_fourbar,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs=_fourbar_scene_kwargs(limits),
    )

    solver = solvers.SolverXPBD(
        model,
        iterations=2,
        soft_body_relaxation=0.9,
        soft_contact_relaxation=0.9,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.4,
        joint_linear_compliance=0.0,
        joint_angular_compliance=0.0,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=True,
        angular_damping=0.0,
        enable_restitution=False,
    )

    return _make_setup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model, q_base),
        friction=0.0,
    )


def make_setup_fourbar_mujoco(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int,
    limits: bool = True,
) -> SolverSetup:
    """Build a non-standalone :class:`SolverSetup` for SolverMuJoCo on the box-fourbar linkage."""
    builder, model, q_base = _build_problem_model(
        solvers.SolverMuJoCo,
        basics.build_boxes_fourbar,
        rigid_contact_max=rigid_contact_max,
        scene_kwargs=_fourbar_scene_kwargs(limits),
    )

    solver = solvers.SolverMuJoCo(
        model,
        cone="elliptic",
        impratio=100,
        iterations=200,
        ls_iterations=100,
        nconmax=rigid_contact_max,
        njmax=100,
        use_mujoco_contacts=False,
    )

    return _make_setup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        dt=dt,
        rigid_contact_max=rigid_contact_max,
        max_log_frames=max_log_frames,
        reset_cb=_make_fk_reset_cb(model, q_base),
        friction=0.0,
    )


###
# Per-problem entry-point factories
###


def build_box_on_plane_run(
    *,
    dt: float,
    max_log_frames: int,
    use_external_force: bool = True,
    rigid_contact_max: int = 46,
) -> ProblemRun:
    """Build the box-on-plane :class:`ProblemRun` (Kamino as leader, XPBD + MuJoCo followers)."""
    kwargs = {"dt": dt, "max_log_frames": max_log_frames, "rigid_contact_max": rigid_contact_max}
    setups = {
        "mujoco": make_setup_box_on_plane_mujoco(**kwargs),
        "xpbd": make_setup_box_on_plane_xpbd(**kwargs),
        "kamino": make_setup_box_on_plane_kamino(**kwargs),
    }
    force_cb = make_external_force_cb_box_on_plane(setups["kamino"]) if use_external_force else None
    return ProblemRun(
        setups=setups,
        force_cb=force_cb,
        camera=(wp.vec3(2.0, 2.0, 0.5), -5.0, 180.0 + 48.0),
    )


def build_fourbar_run(
    *,
    dt: float,
    max_log_frames: int,
    rigid_contact_max: int = 32,
) -> ProblemRun:
    """Build the box-fourbar :class:`ProblemRun` (Kamino as leader, XPBD + MuJoCo followers)."""
    kwargs = {"dt": dt, "max_log_frames": max_log_frames, "rigid_contact_max": rigid_contact_max}
    setups = {
        "mujoco": make_setup_fourbar_mujoco(**kwargs),
        "xpbd": make_setup_fourbar_xpbd(**kwargs),
        "kamino": make_setup_fourbar_kamino(**kwargs),
    }
    return ProblemRun(
        setups=setups,
        force_cb=None,
        camera=(wp.vec3(-0.5, -1.0, 0.2), -5.0, 70.0),
    )
