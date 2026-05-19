# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import warp as wp

from .... import solvers
from ....core import Axis
from ....sim import ModelBuilder
from ....viewer import ViewerBase
from .._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
from .setup import SolverSetup

###
# Module interface
###

# __all__ = ["TODO"]


###
#
###


def make_setup_solver_kamino(asset_file: str, q_base: wp.transformf, dt: float, max_frames: int) -> SolverSetup:
    robot_builder = ModelBuilder(up_axis=Axis.Z)
    solvers.SolverKamino.register_custom_attributes(robot_builder)
    robot_builder.default_shape_cfg.margin = 0.0
    robot_builder.default_shape_cfg.gap = 0.0
    robot_builder.add_usd(
        asset_file,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=True,
        enable_self_collisions=True,
        hide_collision_shapes=True,
    )
    builder = ModelBuilder(up_axis=Axis.Z)
    builder.request_state_attributes("body_parent_f")
    builder.request_contact_attributes("force")
    builder.rigid_contact_max = 32
    builder.add_world(robot_builder)
    builder.add_ground_plane()
    model = builder.finalize(skip_validation_joints=True)
    solver_config = solvers.SolverKamino.Config.from_model(model)
    solver_config.constraints.alpha = 0.01
    solver_config.constraints.beta = 0.01
    solver_config.constraints.gamma = 0.01
    solver_config.constraints.delta = 1e-6
    solver_config.dynamics.preconditioning = True
    solver_config.padmm.primal_tolerance = 1e-6
    solver_config.padmm.dual_tolerance = 1e-6
    solver_config.padmm.compl_tolerance = 1e-6
    solver_config.padmm.max_iterations = 200
    solver_config.padmm.rho_0 = 1.0
    solver_config.compute_solution_metrics = True
    solver = solvers.SolverKamino(model=model, config=solver_config)
    metrics = SolutionMetricsNewton(
        model=builder.finalize(skip_validation_joints=True),
        dt=dt,
        sparse=False,
    )
    logger = SolutionMetricsLogger(
        metrics=metrics,
        max_frames=max_frames,
        mode=SolutionMetricsLogger.Mode.ROLLING,
    )
    logger_solver = SolutionMetricsLogger(
        metrics=solver._solver_kamino.metrics,
        max_frames=max_frames,
        mode=SolutionMetricsLogger.Mode.ROLLING,
    )
    setup = SolverSetup(
        name="kamino",
        builder=builder,
        model=model,
        solver=solver,
        metrics=metrics,
        logger=logger,
        logger_solver=logger_solver,
    )
    base_q = wp.zeros(shape=(1,), dtype=wp.transformf)
    base_q.assign([q_base])
    solver.reset(state_out=setup.state_0, base_q=base_q)
    return setup


def make_setup_solver_mujoco(asset_file: str, q_base: wp.transformf, dt: float, max_frames: int) -> SolverSetup:
    articulation_builder = ModelBuilder(up_axis=Axis.Z)
    solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
    articulation_builder.default_joint_cfg = ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    articulation_builder.default_shape_cfg.ke = 2.0e3
    articulation_builder.default_shape_cfg.kd = 1.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e3
    articulation_builder.default_shape_cfg.mu = 0.75
    articulation_builder.add_usd(
        asset_file,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    articulation_builder.joint_q[:3] = [q_base.p[0], q_base.p[1], q_base.p[2]]
    if len(articulation_builder.joint_q) > 6:
        articulation_builder.joint_q[3:7] = [q_base.q[0], q_base.q[1], q_base.q[2], q_base.q[3]]
    builder = ModelBuilder(up_axis=Axis.Z)
    builder.request_state_attributes("body_parent_f")
    builder.request_contact_attributes("force")
    builder.rigid_contact_max = 32
    builder.add_world(articulation_builder)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    model = builder.finalize()
    solver = solvers.SolverMuJoCo(
        model,
        cone="elliptic",
        impratio=100,
        iterations=200,
        ls_iterations=100,
        nconmax=46,
        njmax=100,
        use_mujoco_contacts=False,
    )
    metrics = SolutionMetricsNewton(
        dt=dt,
        model=model,
        sparse=False,
    )
    logger = SolutionMetricsLogger(
        metrics=metrics,
        max_frames=max_frames,
        mode=SolutionMetricsLogger.Mode.ROLLING,
    )
    setup = SolverSetup(
        name="mujoco",
        builder=builder,
        model=model,
        solver=solver,
        metrics=metrics,
        logger=logger,
    )
    eval_fk(setup.model, setup.model.joint_q, setup.model.joint_qd, setup.state_0)
    return setup


def make_setup_solver_xpbd(asset_file: str, q_base: wp.transformf, dt: float, max_frames: int) -> SolverSetup:
    articulation_builder = ModelBuilder(up_axis=Axis.Z)
    solvers.SolverXPBD.register_custom_attributes(articulation_builder)
    articulation_builder.default_joint_cfg = ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    articulation_builder.default_shape_cfg.ke = 2.0e3
    articulation_builder.default_shape_cfg.kd = 1.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e3
    articulation_builder.default_shape_cfg.mu = 0.75
    articulation_builder.add_usd(
        asset_file,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    articulation_builder.joint_q[:3] = [q_base.p[0], q_base.p[1], q_base.p[2]]
    if len(articulation_builder.joint_q) > 6:
        articulation_builder.joint_q[3:7] = [q_base.q[0], q_base.q[1], q_base.q[2], q_base.q[3]]
    builder = ModelBuilder(up_axis=Axis.Z)
    builder.request_state_attributes("body_parent_f")
    builder.request_contact_attributes("force")
    builder.rigid_contact_max = 32
    builder.add_world(articulation_builder)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    model = builder.finalize()
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
    metrics = SolutionMetricsNewton(
        dt=dt,
        model=model,
        sparse=False,
    )
    logger = SolutionMetricsLogger(
        metrics=metrics,
        max_frames=max_frames,
        mode=SolutionMetricsLogger.Mode.ROLLING,
    )
    setup = SolverSetup(
        name="xpbd",
        builder=builder,
        model=model,
        solver=solver,
        metrics=metrics,
        logger=logger,
    )
    eval_fk(setup.model, setup.model.joint_q, setup.model.joint_qd, setup.state_0)
    return setup
