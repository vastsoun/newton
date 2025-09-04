import os
import time
import h5py

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core.types import float32, vec6f
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.simulation.simulator import Simulator
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.utils.print import print_progress_bar
from newton._src.solvers.kamino.utils.profile import get_device_info
from newton._src.solvers.kamino.models import get_primitives_usd_assets_path
from newton._src.solvers.kamino.models.builders import (
    offset_builder,
    add_ground_geom,
    build_box_pendulum_vertical
)
from newton._src.solvers.kamino.examples import (
    get_examples_data_hdf5_path,
    get_examples_data_npy_path,
    print_frame
)


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

    # Define the time window for the active external force profile
    t_start = float32(1.0)
    t_end = float32(3.0)

    # Get the current time
    t = state_time_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        state_joints_tau_j[0] = 1.0
    else:
        state_joints_tau_j[0] = 0.0


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
USD_MODEL_PATH = os.path.join(get_primitives_usd_assets_path(), "box_pendulum.usda")

# Set the path to the generated HDF5 dataset file
RENDER_DATASET_PATH = os.path.join(get_examples_data_hdf5_path(), "box_pendulum.hdf5")


###
# Main function
###

if __name__ == "__main__":

    # TODO: load these from arguments
    # Application options
    clear_warp_cache = True
    use_cuda_graph = False
    load_from_usd = True
    verbose = False

    # Clear the warp caches
    if clear_warp_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # Warp configs
    # wp.config.verify_fp = True
    # wp.config.verbose = True
    # wp.config.verbose_warnings = True

    # Set global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Get the default warp device
    device = wp.get_preferred_device()
    device = wp.get_device(device)

    # Enable verbose output
    msg.set_log_level(msg.LogLevel.INFO)

    # Determine if using CUDA graphs
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")

    # Create a single-instance system
    msg.info("Setting up model builder...")

    # Create a single-instance system
    if load_from_usd:
        msg.info("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)
    else:
        msg.info("Constructing builder using generator ...")
        builder = ModelBuilder()
        build_box_pendulum_vertical(builder=builder, z_offset=0.7, ground=True)

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

    # Create the output directory if it does not exist
    output_dir = os.path.dirname(RENDER_DATASET_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dataset file and renderer
    msg.info("Creating the HDF5 renderer...")
    datafile = h5py.File(RENDER_DATASET_PATH, 'w')
    renderer = hdf5.DatasetRenderer(sysname="box_pendulum_vertical", datafile=datafile, dt=sim.dt)

    # Store the initial state of the system
    sdata.update_from(simulator=sim)
    cdata.update_from(simulator=sim)
    renderer.add_frame(system=sdata, contacts=cdata)
    if verbose:
        print_frame(sim, 0)

    # Step the simulation and collect frames
    ns = 10000
    msg.info(f"Collecting ns={ns} frames...")
    start_time = time.time()
    with wp.ScopedTimer("sim.step", active=True):
        with wp.ScopedDevice(device):
            for i in range(ns + 1):
                if use_cuda_graph:
                    wp.capture_launch(step_graph)
                else:
                    with wp.ScopedDevice(device):
                        sim.step()
                wp.synchronize()
                sdata.update_from(simulator=sim)
                cdata.update_from(simulator=sim)
                renderer.add_frame(system=sdata, contacts=cdata)
                print_progress_bar(i, ns, start_time, prefix='Progress', suffix='')
                if verbose:
                    print_frame(sim, i + 1)

    # Save the dataset
    msg.info("Saving all frames to HDF5...")
    renderer.save()
