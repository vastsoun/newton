import argparse
import os
import time

import h5py
import numpy as np
import warp as wp

import newton
import newton.examples
import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import float32, vec6f
from newton._src.solvers.kamino.examples import (
    get_examples_data_hdf5_path,
    print_frame
)
from newton._src.solvers.kamino.models import get_primitives_usd_assets_path
from newton._src.solvers.kamino.models.builders import (
    add_ground_geom,
    build_boxes_fourbar
)
from newton._src.solvers.kamino.simulation.simulator import Simulator
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.utils.print import print_progress_bar
from newton._src.solvers.kamino.utils.profile import get_device_info


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
    t_start = float32(2.0)
    t_end = float32(2.5)

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
        ]
    )


###
# Constants
###

# Set the path to the external USD assets
USD_MODEL_PATH = os.path.join(get_primitives_usd_assets_path(), "boxes_fourbar.usda")

# Set the path to the generated HDF5 dataset file
RENDER_DATASET_PATH = os.path.join(get_examples_data_hdf5_path(), "fourbar_free.hdf5")


###
# Main function
###

def run_hdf5_mode(clear_warp_cache=True, use_cuda_graph=False, load_from_usd=False, verbose=False):
    """Run the simulation in HDF5 mode to save data to file."""
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

    # Enable verbose output
    verbose = False
    msg.set_log_level(msg.LogLevel.INFO)

    # Determine if using CUDA graphs
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")

    # Create a single-instance system
    if load_from_usd:
        msg.info("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
    else:
        msg.info("Constructing builder using generator ...")
        builder = ModelBuilder()
        build_boxes_fourbar(builder=builder, z_offset=0.2, ground=False)

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

    # Add a static collision layer and geometry for the plane
    add_ground_geom(builder)

    # Set gravity
    builder.gravity.enabled = True

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

    # Create a dataset file and renderer
    msg.info("Creating the HDF5 renderer...")
    datafile = h5py.File(RENDER_DATASET_PATH, 'w')
    renderer = hdf5.DatasetRenderer(sysname="fourbar_free", datafile=datafile, dt=sim.dt)

    # Store the initial state of the system
    sdata.update_from(simulator=sim)
    cdata.update_from(simulator=sim)
    renderer.add_frame(system=sdata, contacts=cdata)
    if verbose:
        print_frame(sim, 0)

    nbd = sim.model.size.sum_of_num_body_dofs
    njd = sim.model.size.sum_of_num_joint_dofs

    # Step the simulation and collect frames
    ns = 10000
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

                sdata.update_from(simulator=sim)
                cdata.update_from(simulator=sim)
                pdata.update_from(simulator=sim)
                renderer.add_frame(system=sdata, contacts=cdata, problem=pdata)
                print_progress_bar(i, ns, start_time, prefix='Progress', suffix='')

    # Save the dataset
    msg.info("Saving all frames to HDF5...")
    renderer.save()


class BoxesFourbarExample:
    """ViewerGL example class for boxes fourbar simulation."""

    def __init__(self, viewer, load_from_usd=False):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # Get the default warp device
        device = wp.get_preferred_device()
        device = wp.get_device(device)

        # Create a single-instance system
        if load_from_usd:
            msg.info("Constructing builder from imported USD ...")
            importer = USDImporter()
            builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
        else:
            msg.info("Constructing builder using generator ...")
            builder = ModelBuilder()
            build_boxes_fourbar(builder=builder, z_offset=0.2, ground=False)

        # Add a static collision layer and geometry for the plane
        add_ground_geom(builder)

        # Set gravity
        builder.gravity.enabled = True

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=builder, device=device)
        self.sim.set_control_callback(control_callback)

        # Create a simple newton model just for the viewer (with ground plane)
        newton_builder = newton.ModelBuilder()
        newton_builder.add_ground_plane()
        newton_model = newton_builder.finalize()
        self.viewer.set_model(newton_model)

        # Define box dimensions (from build_boxes_fourbar function)
        # Box 1: horizontal bar (d_1=0.1, w_1=0.01, h_1=0.01)
        # Box 2: vertical bar (d_2=0.01, w_2=0.01, h_2=0.1)  
        # Box 3: horizontal bar (d_3=0.1, w_3=0.01, h_3=0.01)
        # Box 4: vertical bar (d_4=0.01, w_4=0.01, h_4=0.1)
        self.box_dimensions = [
            (0.1, 0.01, 0.01),  # Box 1 - horizontal
            (0.01, 0.01, 0.1),  # Box 2 - vertical
            (0.1, 0.01, 0.01),  # Box 3 - horizontal
            (0.01, 0.01, 0.1),  # Box 4 - vertical
        ]

        # Define diverse colors for each box
        self.box_colors = [
            wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
            wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
            wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
            wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
        ]

        # No need for custom ground color - using newton's standard ground plane

        # Initialize the simulator with a warm-up step
        self.sim.reset()
        
        # Don't capture graphs initially to avoid CUDA stream conflicts
        self.graph = None

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
            
            # Debug: print body poses info for first few frames
            if self.sim_time < 0.1:
                print(f"Number of bodies: {len(body_poses)}")
                print(f"Body poses shape: {body_poses.shape}")
                if len(body_poses) > 0:
                    print(f"First body pose: {body_poses[0]}")

            # Render each box using log_shapes
            for i, (dimensions, color) in enumerate(zip(self.box_dimensions, self.box_colors)):
                if i < len(body_poses):
                    # Convert kamino transformf to warp transform
                    pose = body_poses[i]
                    # kamino transformf has [x, y, z, qx, qy, qz, qw] format
                    position = wp.vec3(float(pose[0]), float(pose[1]), float(pose[2]))
                    quaternion = wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                    transform = wp.transform(position, quaternion)

                    # Log the box shape
                    self.viewer.log_shapes(
                        f"/fourbar/box_{i+1}",
                        newton.GeoType.BOX,
                        dimensions,
                        wp.array([transform], dtype=wp.transform),
                        color,
                    )
                    
                    # Debug: print transform info for first few frames
                    if self.sim_time < 0.1:
                        print(f"Box {i+1}: pos={position}, quat={quaternion}")

        except Exception as e:
            print(f"Error accessing body poses: {e}")
            print(f"Available attributes: {dir(self.sim.model_data.bodies)}")

        self.viewer.end_frame()

    def test(self):
        """Test function for compatibility."""
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxes fourbar simulation example")
    parser.add_argument(
        "--mode",
        choices=["hdf5", "viewer"],
        default="viewer",
        help="Simulation mode: 'hdf5' for data collection, 'viewer' for live visualization"
    )
    parser.add_argument("--clear-cache", action="store_true", default=True, help="Clear warp cache")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA graphs")
    parser.add_argument("--load-from-usd", action="store_true", help="Load model from USD file")
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
        run_hdf5_mode(
            clear_warp_cache=args.clear_cache,
            use_cuda_graph=args.cuda_graph,
            load_from_usd=args.load_from_usd,
            verbose=args.verbose
        )
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
        example = BoxesFourbarExample(viewer, load_from_usd=args.load_from_usd)
        newton.examples.run(example)
