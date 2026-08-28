# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stack cubes with a UR5e and a closed-loop Robotiq 2F-85 in Kamino.

Command: python -m newton.examples kamino_ur5e_robotiq_stacking --world-count 16
or: PYGLET_HEADLESS=1 uv run -m newton.examples kamino_ur5e_robotiq_stacking --world-count 16 \
    --viewer gl --headless --num-frames 2000 --video-path ur5e_robotiq_stacking.mp4
"""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton._src.utils.download_assets import MENAGERIE_REF, MENAGERIE_URL
from newton.examples.ik.example_ik_cube_stacking import (
    Example as CubeStackingExample,
    advance_task_kernel,
    set_target_pose_kernel,
)


class Example(CubeStackingExample):
    """Run a three-cube Kamino stacking task with a UR5e and Robotiq gripper."""

    def __init__(self, viewer, args):
        args.solver = "kamino"
        super().__init__(viewer, args)

    def _download_menagerie_model(self, folder: str, filename: str):
        """Download a pinned MuJoCo Menagerie model."""
        return newton.examples.download_external_git_folder(MENAGERIE_URL, folder, ref=MENAGERIE_REF) / filename

    @staticmethod
    def _body_index(builder: newton.ModelBuilder, suffix: str) -> int:
        """Return the uniquely named imported body ending with suffix."""
        return next(i for i, label in enumerate(builder.body_label) if label.endswith(suffix))

    @staticmethod
    def _add_loop_joint(builder: newton.ModelBuilder, parent: int, child: int, label: str):
        """Close a four-bar linkage at the child body's local origin."""
        world_anchor = wp.transform_point(builder.body_q[child], wp.vec3())
        parent_anchor = wp.transform_point(wp.transform_inverse(builder.body_q[parent]), world_anchor)
        child_anchor = wp.transform_point(wp.transform_inverse(builder.body_q[child]), world_anchor)
        builder.add_joint_ball(
            parent=parent,
            child=child,
            parent_xform=wp.transform(parent_anchor, wp.quat_identity()),
            child_xform=wp.transform(child_anchor, wp.quat_identity()),
            label=label,
        )
        return parent_anchor, child_anchor

    def _add_native_robotiq_loops(self, builder: newton.ModelBuilder) -> None:
        """Replace the Menagerie equality connects with Kamino-native loop joints."""
        self.gripper_loop_anchors = []
        for side in ("right", "left"):
            parent = self._body_index(builder, f"/{side}_coupler")
            child = self._body_index(builder, f"/{side}_follower")
            parent_anchor, child_anchor = self._add_loop_joint(
                builder,
                parent,
                child,
                f"robotiq_{side}_fourbar_loop",
            )
            self.gripper_loop_anchors.append((parent, child, parent_anchor, child_anchor))

    @staticmethod
    def _disable_robotiq_link_collisions(builder: newton.ModelBuilder) -> None:
        """Keep the pad boxes as the Robotiq's only colliders."""
        for shape, label in enumerate(builder.shape_label):
            is_robotiq_mesh = label.startswith("robotiq_2f85/") and builder.shape_type[shape] == newton.GeoType.MESH
            if is_robotiq_mesh:
                builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_SHAPES

    def _configure_arm_targets(self, builder: newton.ModelBuilder) -> None:
        """Configure position control for the six UR5e arm coordinates."""
        self.arm_dof_count = builder.joint_dof_count
        home = [0.0, -wp.half_pi, wp.half_pi, -wp.half_pi, -wp.half_pi, 0.0]
        builder.joint_q[: self.arm_dof_count] = home
        builder.joint_target_q[: self.arm_dof_count] = home
        builder.joint_target_ke[: self.arm_dof_count] = [4500.0] * self.arm_dof_count
        builder.joint_target_kd[: self.arm_dof_count] = [450.0] * self.arm_dof_count
        builder.joint_effort_limit[: self.arm_dof_count] = [100.0] * self.arm_dof_count
        builder.joint_armature[: self.arm_dof_count] = [0.2] * self.arm_dof_count
        builder.joint_target_mode[: self.arm_dof_count] = [int(newton.JointTargetMode.POSITION)] * self.arm_dof_count

    def build_franka_with_table(self):
        """Build the UR5e, a native-loop Robotiq gripper, and the task table."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.request_contact_attributes("force")

        self.ur5e_path = self._download_menagerie_model("universal_robots_ur5e", "ur5e.xml")
        self.robotiq_path = self._download_menagerie_model("robotiq_2f85", "2f85.xml")
        base_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        builder.add_mjcf(
            str(self.ur5e_path),
            xform=wp.transform(self.robot_base_pos, base_rotation),
            floating=False,
            collapse_fixed_joints=False,
        )
        self._configure_arm_targets(builder)

        wrist = self._body_index(builder, "/wrist_3_link")
        self.ee_index = wrist
        gripper_mount_xform = wp.transform(
            wp.vec3(0.0, 0.1, 0.0),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2.0),
        )
        self.gripper_dof_offset = builder.joint_dof_count
        builder.add_mjcf(
            str(self.robotiq_path),
            xform=gripper_mount_xform,
            parent_body=wrist,
            skip_equality_constraints=True,
            collapse_fixed_joints=False,
        )
        self._disable_robotiq_link_collisions(builder)
        pinch_site = next(i for i, label in enumerate(builder.shape_label) if label.endswith("/base/pinch"))
        pinch_body = builder.shape_body[pinch_site]
        pinch_world_xform = wp.transform_multiply(builder.body_q[pinch_body], builder.shape_transform[pinch_site])
        self.ee_local_xform = wp.transform_multiply(wp.transform_inverse(builder.body_q[wrist]), pinch_world_xform)
        self._add_native_robotiq_loops(builder)
        self.gripper_driver_dofs = [self.gripper_dof_offset, self.gripper_dof_offset + 4]
        for dof in self.gripper_driver_dofs:
            builder.joint_q[dof] = 0.0
            builder.joint_target_q[dof] = 0.0
            builder.joint_target_ke[dof] = 100.0
            builder.joint_target_kd[dof] = 5.0
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)

        cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0, density=1000.0, mu=0.9)
        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=cfg,
            label="table",
        )
        return builder

    def setup_ik(self):
        """Create a six-coordinate UR5e IK problem independent of gripper loops."""
        arm_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        arm_builder.add_mjcf(
            str(self.ur5e_path),
            xform=wp.transform(
                self.robot_base_pos,
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
            ),
            floating=False,
            collapse_fixed_joints=False,
        )
        self._configure_arm_targets(arm_builder)
        self.arm_model = arm_builder.finalize()
        self.arm_ee_index = self._body_index(arm_builder, "/wrist_3_link")

        body_q = self.state.body_q.numpy()
        self.ee_tf = wp.transform_multiply(wp.transform(*body_q[self.ee_index]), self.ee_local_xform)
        self.home_pos = wp.transform_get_translation(self.ee_tf)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.arm_ee_index,
            link_offset=wp.transform_get_translation(self.ee_local_xform),
            target_positions=wp.array([self.home_pos] * self.world_count, dtype=wp.vec3),
        )
        rotation = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.arm_ee_index,
            link_offset_rotation=wp.transform_get_rotation(self.ee_local_xform),
            target_rotations=wp.array([rotation[:4]] * self.world_count, dtype=wp.vec4),
        )
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.arm_model.joint_limit_lower,
            joint_limit_upper=self.arm_model.joint_limit_upper,
        )
        self.joint_q_ik = wp.array([self.arm_model.joint_q.numpy()] * self.world_count, dtype=wp.float32)
        self.ik_iters = 32
        self.ik_solver = ik.IKSolver(
            model=self.arm_model,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def setup_tasks(self):
        """Allocate the inherited cube-stacking schedule and Robotiq targets."""
        super().setup_tasks()
        self.task_position_tolerance = 0.002
        self.gripper_open_position = 0.0
        self.gripper_closed_position = 0.8
        self.gripper_target_interpolated = wp.zeros(shape=(self.world_count, 2), dtype=wp.float32)

    def capture_ik(self):
        """Disable a separate IK graph because the arm-only model is host initialized."""
        self.graph_ik = None

    def set_joint_targets(self):
        """Solve arm IK and command both driver joints to keep the gripper symmetric."""
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_schedule,
                self.task_time_soft_limits,
                self.task_object,
                self.task_idx,
                self.task_time_elapsed,
                self.task_dt,
                self.task_offset_approach,
                self.task_offset_lift,
                self.task_offset_retract,
                self.task_drop_off_pos,
                self.cube_size,
                self.gripper_open_position,
                self.gripper_closed_position,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.ee_local_xform,
                self.robot_body_count,
                self.num_bodies_per_world,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_target_interpolated,
                self.ee_rot_target,
                self.ee_rot_target_interpolated,
                self.gripper_target_interpolated,
            ],
        )
        self.pos_obj.set_target_positions(self.ee_pos_target_interpolated)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interpolated)
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        targets = self.control.joint_target_q.reshape((self.world_count, -1))
        wp.copy(dest=targets[:, : self.arm_dof_count], src=self.joint_q_ik)
        for driver in self.gripper_driver_dofs:
            wp.copy(dest=targets[:, driver], src=self.gripper_target_interpolated[:, 0])
        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_time_soft_limits,
                self.task_position_tolerance,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.num_bodies_per_world,
                self.ee_index,
                self.ee_local_xform,
            ],
            outputs=[self.task_idx, self.task_time_elapsed, self.task_init_body_q],
        )

    def test_final(self):
        """Verify finite native-loop state and completed stacking when applicable."""
        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise ValueError("Robotiq closed-loop simulation produced non-finite body poses")

        for world_id in range(self.world_count):
            offset = world_id * self.num_bodies_per_world
            for parent, child, parent_anchor, child_anchor in self.gripper_loop_anchors:
                parent_world = wp.transform_point(wp.transform(*body_q[offset + parent]), parent_anchor)
                child_world = wp.transform_point(wp.transform(*body_q[offset + child]), child_anchor)
                if np.linalg.norm(np.asarray(parent_world) - np.asarray(child_world)) > 2.0e-3:
                    raise ValueError("Robotiq four-bar loop anchor drift exceeded 2 mm")

        targets = self.control.joint_target_q.numpy().reshape((self.world_count, -1))
        if not np.allclose(
            targets[:, self.gripper_driver_dofs[0]],
            targets[:, self.gripper_driver_dofs[1]],
        ):
            raise ValueError("Robotiq driver joint targets are not synchronized")

        if np.all(self.task_idx.numpy() == self.task_counter - 1):
            super().test_final()

    @staticmethod
    def create_parser():
        """Create command-line arguments for the Kamino-only example."""
        parser = CubeStackingExample.create_parser()
        parser.set_defaults(solver="kamino", world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
