import argparse
import os
import time

import h5py
import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.utils.logger as msg
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import float32, vec6f

# Note: Keeping imports for potential use in commented code sections
# from newton._src.solvers.kamino.models.builders import (
#     add_ground_geom,
#     add_velocity_bias,
#     offset_builder,
# )
from newton._src.solvers.kamino.examples import get_examples_data_hdf5_path, get_examples_data_root_path, print_frame
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.simulation.simulator import Simulator
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.utils.print import print_progress_bar
from newton._src.solvers.kamino.utils.profile import get_device_info

# Note: Keeping import for potential use in commented code sections
# from newton._src.solvers.kamino.tests.test_solvers_padmm import save_solver_info


###
# Kernels
###


@wp.kernel
def _control_callback(
    model_time_dt: wp.array(dtype=float32),
    state_time_t: wp.array(dtype=float32),
    state_joints_q_j: wp.array(dtype=float32),
    state_joints_dq_j: wp.array(dtype=float32),
    state_joints_tau_j: wp.array(dtype=float32),
    state_bodies_w_e_i: wp.array(dtype=vec6f),
):
    """
    An example control callback kernel.
    """
    # Set world index
    wid = int(0)
    jid = int(0)

    # Define the time window for the active external force profile
    t_start = float32(0.0)
    t_end = float32(2.0)

    # Get the current time
    t = state_time_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        state_joints_tau_j[jid] = 0.1
    else:
        state_joints_tau_j[jid] = 0.0


###
# Launchers
###


def control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _control_callback,
        dim=1,
        inputs=[
            sim.model.time.dt,
            sim.model_data.time.time,
            sim.model_data.joints.q_j,
            sim.model_data.joints.dq_j,
            sim.model_data.joints.tau_j,
            sim.model_data.bodies.w_e_i,
        ],
    )


###
# Constants
###

# Set the path to the external USD assets
USD_MODEL_PATH = os.path.join(get_examples_usd_assets_path(), "walker/walker_floating_with_boxes.usda")

# Set the path to the generated HDF5 dataset file
RENDER_DATASET_PATH = os.path.join(get_examples_data_hdf5_path(), "walker.hdf5")

# Set the path to the generated numpy dataset file
PADMM_INFO_PATH = os.path.join(get_examples_data_root_path(), "padmm")


###
# Main function
###


def run_hdf5_mode(clear_warp_cache=True, use_cuda_graph=False, verbose=False):
    """Run the simulation in HDF5 mode to save data to file."""
    # Application options

    # Clear the warp caches
    if clear_warp_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # Warp configs
    # wp.config.verify_fp = True
    # wp.config.verbose = True
    # wp.config.verbose_warnings = True

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Get the default warp device
    device = wp.get_preferred_device()
    device = wp.get_device(device)
    msg.info(f"device: {device}")

    # Enable verbose output
    msg.set_log_level(msg.LogLevel.INFO)

    # Determine if using CUDA graphs
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")

    # Create a single-instance system
    msg.info("Constructing builder from imported USD ...")
    importer = USDImporter()
    builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)

    # # Print builder data
    # msg.info("\n\nbody poses:")
    # for body in builder.bodies:
    #     msg.info(f"  {body.name}: q_i_0: {body.q_i_0}")
    # msg.info("\n\njoint params:")
    # for joint in builder.joints:
    #     msg.info(f"  {joint.name}: B_r_Bj: {joint.B_r_Bj}")
    #     msg.info(f"  {joint.name}: F_r_Fj: {joint.F_r_Fj}")
    #     msg.info(f"  {joint.name}: X_j:\n{joint.X_j}")
    # msg.info("\n\ncgeom params:")
    # for geom in builder.collision_geoms:
    #     msg.info(f"  {geom.name}: offset: {geom.offset}")

    # # Offset the model to place it above the ground
    # # NOTE: The USD model is centered at the origin
    # offset = wp.transformf(0.0, 0.0, 0.28, 0.0, 0.0, 0.0, 1.0)
    # offset_builder(builder=builder, offset=offset)

    # # Add a static collision layer and geometry for the plane
    # add_ground_geom(builder, group=2, collides=3)

    # # Apply an offset to the whole model
    # r_offset = vec3f(0.0, 0.0, 0.0)
    # R_offset = R_z(1.0) @ R_y(1.0) @ R_x(1.0)
    # q_offset = wp.quat_from_matrix(R_offset)
    # offset = transformf(r_offset, q_offset)
    # offset_builder(builder=builder, offset=offset)

    # v_bias = vec3f(0.0, 0.0, 0.0)
    # omega_bias = vec3f(0.0, 0.0, 1.0)
    # u_bias = screw(v_bias, omega_bias)
    # add_velocity_bias(builder=builder, bias=u_bias)

    # Set gravity
    builder.gravity.enabled = False

    # Print the collision masking of each geom
    for i in range(len(builder.collision_geoms)):
        msg.info(
            f"builder.cgeom{i}: group={builder.collision_geoms[i].group}, collides={builder.collision_geoms[i].collides}"
        )

    # Create a simulator
    msg.info("Building the simulator...")
    sim = Simulator(builder=builder, device=device)
    sim.set_control_callback(control_callback)

    # Capture graphs for simulator ops: reset and step
    use_cuda_graph &= can_use_cuda_graph
    reset_graph = None
    step_graph = None
    if use_cuda_graph:
        with wp.ScopedCapture(device) as reset_capture:
            sim.reset()
        reset_graph = reset_capture.graph
        with wp.ScopedCapture(device) as step_capture:
            sim.step()
        step_graph = step_capture.graph

    # Warm-start the simulator before rendering
    # NOTE: This compiles and loads the warp kernels prior to execution
    msg.info("Warming up the simulator...")
    if use_cuda_graph:
        print("Running with CUDA graphs...")
        wp.capture_launch(reset_graph)
        wp.capture_launch(step_graph)
    else:
        msg.info("Running with kernels...")
        with wp.ScopedDevice(device):
            sim.step()
            sim.reset()

    # Print application info
    msg.info("%s", get_device_info(device))

    # Construct and configure the data containers
    msg.info("Setting up HDF5 data containers...")
    sdata = hdf5.RigidBodySystemData()
    sdata.configure(simulator=sim)
    cdata = hdf5.ContactsData()
    pdata = hdf5.DualProblemData()
    pdata.configure(simulator=sim)

    # Create the output directory if it does not exist
    render_dir = os.path.dirname(RENDER_DATASET_PATH)
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    # Create the directory for PADMM info
    if not os.path.exists(PADMM_INFO_PATH):
        os.makedirs(PADMM_INFO_PATH)

    # Create a dataset file and renderer
    msg.info("Creating the HDF5 renderer...")
    datafile = h5py.File(RENDER_DATASET_PATH, "w")
    renderer = hdf5.DatasetRenderer(sysname="walker", datafile=datafile, dt=sim.dt)

    # Store the initial state of the system
    sdata.update_from(simulator=sim)
    cdata.update_from(simulator=sim)
    renderer.add_frame(system=sdata, contacts=cdata)
    if verbose:
        print_frame(sim, 0)

    nbd = sim.model.size.sum_of_num_body_dofs
    njd = sim.model.size.sum_of_num_joint_dofs

    # Step the simulation and collect frames
    ns = 3000  # TODO: 25000
    msg.info(f"Collecting ns={ns} frames...")
    start_time = time.time()
    with wp.ScopedTimer("sim.step", active=True):
        with wp.ScopedDevice(device):
            for i in range(ns):
                if use_cuda_graph:
                    wp.capture_launch(step_graph)
                else:
                    with wp.ScopedDevice(device):
                        sim.step()
                wp.synchronize()

                status = sim._dual_solver.data.status.numpy()
                msg.warning(f"[{i}]: solver.iterations : {status[0][1]}")
                # msg.warning(f"[{i}]: nl: {sim.model_data.info.num_limits.numpy()[0]}")
                # msg.warning(f"[{i}]: nc: {sim.model_data.info.num_contacts.numpy()[0]}")
                # save_solver_info(sim._dual_solver, path=os.path.join(PADMM_INFO_PATH, f"padmm_solver_info_{i}.pdf"))

                # msg.warning(f"cgeoms.offset :\n{sim.model.cgeoms.offset}")
                # msg.warning(f"cgeoms.pose :\n{sim.model_data.cgeoms.pose}")
                # msg.warning(f"collisions.model_num_collisions :\n{sim.collision_detector.collisions.cdata.model_num_collisions}")
                # msg.warning(f"collisions.geom_pair :\n{sim.collision_detector.collisions.cdata.geom_pair}")
                # msg.warning(f"contacts.model_num_collisions :\n{sim.contacts.data.model_num_contacts}")
                # msg.warning(f"contacts.gapfunc :\n{sim.contacts.data.gapfunc}")

                # maxdim_np = sim._dual_problem.data.maxdim.numpy()
                # dim_np = sim._dual_problem.data.dim.numpy()
                # v_f_np = sim._dual_problem.data.v_f.numpy()
                # D_np = sim._dual_problem.delassus.data.D.numpy()
                # L_np = sim._dual_problem.delassus.factorizer.data.L.numpy()
                # y_np = sim._dual_problem.delassus.factorizer.data.y.numpy()
                lambda_np = sim._dual_solver.data.solution.lambdas.numpy()
                u_np = sim.model_data.bodies.u_i.numpy()
                q_np = sim.model_data.bodies.q_i.numpy()

                # is_v_f_finite = np.isfinite(v_f_np).all()
                # is_D_finite = np.isfinite(D_np).all()
                # is_L_finite = np.isfinite(L_np).all()
                # is_y_finite = np.isfinite(y_np).all()
                is_lambda_valid = np.all(np.abs(lambda_np) < 1e1)
                is_u_valid = np.all(np.abs(u_np) < 1e2)
                is_q_valid = np.all(np.abs(q_np) < 1e2)
                is_q_finite = np.isfinite(q_np).all()
                # if not is_v_f_finite:
                #     msg.error("v_f is not finite!")
                # if not is_D_finite:
                #     msg.error("D is not finite!")
                # if not is_L_finite:
                #     msg.error("L is not finite!")
                # if not is_y_finite:
                #     msg.error("y is not finite!")

                sdata.update_from(simulator=sim)
                cdata.update_from(simulator=sim)
                pdata.update_from(simulator=sim)
                renderer.add_frame(system=sdata, contacts=cdata, problem=pdata)
                print_progress_bar(i, ns, start_time, prefix="Progress", suffix="")
                # if verbose:
                #     print_frame(sim, i + 1)

                if not is_lambda_valid:
                    msg.error("lambda is not valid!")
                    break

                if not is_u_valid:
                    msg.error("u_i is not valid!")
                    break

                if not is_q_valid:
                    msg.error("q_i is not valid!")
                    break

                if not is_q_finite:
                    msg.error("q_i is not finite!")
                    break

    # Save the dataset
    msg.info("Saving all frames to HDF5...")
    renderer.save()


class WalkerExample:
    """ViewerGL example class for walker simulation."""

    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # Get the default warp device
        device = wp.get_preferred_device()
        device = wp.get_device(device)

        # Create a single-instance system (always load from USD for walker)
        msg.info("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)

        # Set gravity (disabled for walker as in original)
        builder.gravity.enabled = False

        # Print the collision masking of each geom (preserve original logging)
        for i in range(len(builder.collision_geoms)):
            msg.info(
                f"builder.cgeom{i}: group={builder.collision_geoms[i].group}, collides={builder.collision_geoms[i].collides}"
            )

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=builder, device=device)
        self.sim.set_control_callback(control_callback)

        # Don't set a newton model - we'll render everything manually using log_shapes
        self.viewer.set_model(None)

        # Extract geometry information from the kamino simulator
        self.extract_geometry_info()

        # Define colors for different parts of the walker
        self.body_colors = [
            wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
            wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
            wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
            wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
            wp.array([wp.vec3(0.2, 0.8, 0.2)], dtype=wp.vec3),  # Green
            wp.array([wp.vec3(0.8, 0.8, 0.2)], dtype=wp.vec3),  # Yellow
            wp.array([wp.vec3(0.8, 0.2, 0.8)], dtype=wp.vec3),  # Magenta
            wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),  # Gray
        ]

        # Initialize the simulator with a warm-up step
        self.sim.reset()

        # Don't capture graphs initially to avoid CUDA stream conflicts
        self.graph = None

    def extract_geometry_info(self):
        """Extract geometry information from the kamino simulator."""
        # Get collision geometry information from the simulator
        cgeom_model = self.sim.model.cgeoms

        self.geometry_info = []

        # Extract geometry info from collision geometries
        for i in range(cgeom_model.num_geoms):
            bid = cgeom_model.bid.numpy()[i]  # Body ID (-1 for static/ground)
            sid = cgeom_model.sid.numpy()[i]  # Shape ID
            params = cgeom_model.params.numpy()[i]  # Shape parameters
            offset = cgeom_model.offset.numpy()[i]  # Geometry offset

            # Store geometry information for rendering
            geom_info = {"body_id": bid, "shape_id": sid, "params": params, "offset": offset}
            self.geometry_info.append(geom_info)

    def capture(self):
        """Capture CUDA graph if available."""
        # For now, disable CUDA graph capture to avoid stream conflicts
        # This can be re-enabled later if needed with proper stream management
        self.graph = None

    def simulate(self):
        """Run simulation substeps."""
        for _ in range(self.sim_substeps):
            self.sim.step()

    def step(self):
        """Step the simulation."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current frame."""
        self.viewer.begin_frame(self.sim_time)

        # Extract body poses from the kamino simulator
        try:
            body_poses = self.sim.model_data.bodies.q_i.numpy()

            # Render each geometry using log_shapes
            for i, geom_info in enumerate(self.geometry_info):
                bid = geom_info["body_id"]
                sid = geom_info["shape_id"]
                params = geom_info["params"]
                offset = geom_info["offset"]

                # Skip static geometries (ground plane, etc.)
                if bid == -1:
                    continue

                # Get body pose if available
                if bid < len(body_poses):
                    # Convert kamino transformf to warp transform
                    pose = body_poses[bid]
                    # kamino transformf has [x, y, z, qx, qy, qz, qw] format
                    position = wp.vec3(float(pose[0]), float(pose[1]), float(pose[2]))
                    quaternion = wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                    body_transform = wp.transform(position, quaternion)

                    # Apply geometry offset
                    offset_pos = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                    offset_quat = wp.quat(float(offset[3]), float(offset[4]), float(offset[5]), float(offset[6]))
                    offset_transform = wp.transform(offset_pos, offset_quat)

                    # Combine body and offset transforms
                    final_transform = wp.transform_multiply(body_transform, offset_transform)

                    # Choose color based on body ID
                    color_idx = bid % len(self.body_colors)
                    color = self.body_colors[color_idx]

                    # Render based on shape type
                    if sid == 5:  # BOX shape (SHAPE_BOX = 5)
                        # Convert kamino full dimensions to newton half-extents
                        half_extents = (params[0] / 2, params[1] / 2, params[2] / 2)

                        self.viewer.log_shapes(
                            f"/walker/body_{bid}_geom_{i}",
                            newton.GeoType.BOX,
                            half_extents,
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )
                    elif sid == 1:  # SPHERE shape (SHAPE_SPHERE = 1)
                        radius = params[0]

                        self.viewer.log_shapes(
                            f"/walker/body_{bid}_geom_{i}",
                            newton.GeoType.SPHERE,
                            radius,
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )
                    elif sid == 2:  # CAPSULE shape (SHAPE_CAPSULE = 2)
                        radius = params[0]
                        half_height = params[1] / 2

                        self.viewer.log_shapes(
                            f"/walker/body_{bid}_geom_{i}",
                            newton.GeoType.CAPSULE,
                            (radius, half_height),
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )

        except Exception as e:
            print(f"Error accessing body poses: {e}")
            print(f"Available attributes: {dir(self.sim.model_data.bodies)}")

        self.viewer.end_frame()

    def test(self):
        """Test function for compatibility."""
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walker simulation example")
    parser.add_argument(
        "--mode",
        choices=["hdf5", "viewer"],
        default="viewer",
        help="Simulation mode: 'hdf5' for data collection, 'viewer' for live visualization",
    )
    parser.add_argument("--clear-cache", action="store_true", default=True, help="Clear warp cache")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA graphs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Add viewer arguments when in viewer mode
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun", "null"], default="gl", help="Viewer type")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, help="Compute device")
    parser.add_argument("--output-path", type=str, help="Output path for USD viewer")
    parser.add_argument("--num-frames", type=int, default=1000, help="Number of frames for null/USD viewer")

    args = parser.parse_args()

    if args.mode == "hdf5":
        msg.info("Running in HDF5 mode...")
        run_hdf5_mode(clear_warp_cache=args.clear_cache, use_cuda_graph=args.cuda_graph, verbose=args.verbose)
    elif args.mode == "viewer":
        msg.info("Running in ViewerGL mode...")

        # Set device if specified
        if args.device:
            wp.set_device(args.device)

        # Create viewer based on type
        if args.viewer == "gl":
            viewer = newton.viewer.ViewerGL(headless=args.headless)
        elif args.viewer == "usd":
            if args.output_path is None:
                raise ValueError("--output-path is required when using usd viewer")
            viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
        elif args.viewer == "rerun":
            viewer = newton.viewer.ViewerRerun()
        elif args.viewer == "null":
            viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        else:
            raise ValueError(f"Invalid viewer: {args.viewer}")

        # Create and run example
        example = WalkerExample(viewer)

        # Set initial camera position for better view of the walker
        if hasattr(viewer, "set_camera"):
            # Position camera to get a good view of the walker
            camera_pos = wp.vec3(2.0, -3.0, 1.5)
            pitch = -20.0
            yaw = 130.0
            viewer.set_camera(camera_pos, pitch, yaw)

        newton.examples.run(example)
