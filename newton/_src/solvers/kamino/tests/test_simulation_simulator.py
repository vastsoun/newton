###########################################################################
# KAMINO: UNIT TESTS
###########################################################################

import unittest
import numpy as np
import warp as wp

from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_hinged,
    build_boxes_nunchaku,
)
from newton._src.solvers.kamino.models.utils import (
    make_single_builder,
    make_homogeneous_builder,
    make_heterogeneous_builder
)

# Module to be tested
from newton._src.solvers.kamino.simulation.simulator import Simulator


###
# Tests
###

class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.verbose = False  # Set to True for detailed output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_make_simulator(self):
        # Construct the model description using model builders for different systems
        # builder, num_bodies, num_jcts = make_single_builder(build_func=build_box_on_plane)
        # builder, num_bodies, num_jcts = make_single_builder(build_func=build_boxes_nunchaku)
        # builder, _, _ = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_box_on_plane)
        # builder, _, _ = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_boxes_nunchaku)
        builder, num_bodies, num_jcts = make_heterogeneous_builder()

        # Create a simulator
        sim = Simulator(builder=builder, device=self.default_device)

        # Optional verbose output
        if self.verbose:
            print(f"sim.model.size:\n{sim.model.size}")

        # Check the the simulators memory allocations
        # TODO: What else to test here?
        self.assertEqual(sim.model.size.sum_of_num_bodies, sum(num_bodies))
        self.assertEqual(sim.model.size.sum_of_num_joint_cts, sum(num_jcts))

    def test_simulator_reset(self):
        # Construct the model description using model builders for different systems
        builder, _, _ = make_heterogeneous_builder()

        # Create a simulator
        sim = Simulator(builder=builder, device=self.default_device)

        # Rest the simulator
        with wp.ScopedTimer("reset"):
            sim.reset()

        # Optional verbose output
        if self.verbose:
            print(f"[before]: sim._data.state.bodies.w_i:\n{sim._data.state.bodies.q_i}")
            print(f"[after]:  sim._data.state.bodies.w_i:\n{sim._data.state.bodies.q_i}")
            print(f"[before]: sim._data.state.bodies.I_i:\n{sim._data.state.bodies.u_i}")
            print(f"[after]:  sim._data.state.bodies.I_i:\n{sim._data.state.bodies.u_i}")

        # TODO: What to test here?

    def test_simulator_forward(self):
        # Construct the model description using model builders for different systems
        builder, _, _ = make_heterogeneous_builder()

        # Create a simulator
        sim = Simulator(builder=builder, device=self.default_device)

        # Rest the simulator
        if self.verbose:
            print(f"[before]: sim.model.summary.num_bodies: {sim.model.size.sum_of_num_bodies}")
            print(f"[before]: sim._data.state.bodies.w_i:\n{sim._data.state.bodies.w_i}")
            print(f"[before]: sim._data.state.bodies.I_i:\n{sim._data.state.bodies.I_i}")
            print(f"[before]: sim._data.state.bodies.inv_I_i:\n{sim._data.state.bodies.inv_I_i}\n")
        sim._forward()
        if self.verbose:
            print(f"[after]: sim._data.state.bodies.w_i:\n{sim._data.state.bodies.w_i}")
            print(f"[after]: sim._data.state.bodies.I_i:\n{sim._data.state.bodies.I_i}")
            print(f"[after]: sim._data.state.bodies.inv_I_i:\n{sim._data.state.bodies.inv_I_i}\n")

    def test_simulator_advance_time(self):
        # Construct the model description using model builders for different systems
        builder, _, _ = make_heterogeneous_builder()

        # Create a simulator
        sim = Simulator(builder=builder, device=self.default_device)

        # Rest the simulator
        if self.verbose:
            print(f"[before]: sim._data.state.time.steps: {sim._data.state.time.steps}")
            print(f"[before]: sim._data.state.time.time: {sim._data.state.time.time}")
        sim._advance_time()
        sim._advance_time()
        sim._advance_time()
        if self.verbose:
            print(f"[after]: sim._data.state.time.steps: {sim._data.state.time.steps}")
            print(f"[after]: sim._data.state.time.time: {sim._data.state.time.time}")

        # Capture and check the time state
        steps = sim._data.state.time.steps.numpy()
        times = sim._data.state.time.time.numpy()
        for i in range(1, sim.model.size.num_worlds):
            self.assertEqual(steps[i], np.int32(3))
            self.assertEqual(times[i], np.float32(0.003))

    def test_step_simulator(self):
        # Constants
        ns = 1

        # Construct the model description using model builders for different systems
        builder, _, _ = make_heterogeneous_builder()

        # Create a simulator
        sim = Simulator(builder=builder, device=self.default_device)

        # Rest the simulator
        if self.verbose:
            print(f"[pre-step]: sim._data.state.time.steps: {sim._data.state.time.steps}")
            print(f"[pre-step]: sim._data.state.time.time: {sim._data.state.time.time}")
            print(f"[pre-step]: sim._data.state.bodies.q_i:\n{sim._data.state.bodies.q_i}")
            print(f"[pre-step]: sim._data.state.bodies.u_i:\n{sim._data.state.bodies.u_i}")
        with wp.ScopedTimer("step", active=self.verbose):
            for _ in range(ns):
                sim.step()

        if self.verbose:
            print(f"[post-step]: sim._data.s_n.q_i:\n{sim._data.s_n.q_i}")
            print(f"[post-step]: sim._data.s_n.u_i:\n{sim._data.s_n.u_i}")
            print(f"[post-step]: sim._data.state.time.steps: {sim._data.state.time.steps}")
            print(f"[post-step]: sim._data.state.time.time: {sim._data.state.time.time}")
            print(f"[post-step]: sim._data.state.bodies.q_i:\n{sim._data.state.bodies.q_i}")
            print(f"[post-step]: sim._data.state.bodies.u_i:\n{sim._data.state.bodies.u_i}")
            print(f"[post-step]: sim.collision_detector.collisions.cdata.model_num_collisions: {sim.collision_detector.collisions.cdata.model_num_collisions}")
            print(f"[post-step]: sim.collision_detector.collisions.cdata.world_num_collisions: {sim.collision_detector.collisions.cdata.world_num_collisions}")
            print(f"[post-step]: sim.collision_detector.collisions.cdata.wid: {sim.collision_detector.collisions.cdata.wid}")
            print(f"[post-step]: sim.collision_detector.collisions.cdata.geom_pair:\n{sim.collision_detector.collisions.cdata.geom_pair}")
            print(f"[post-step]: sim.collision_detector.contacts.model_max_contacts: {sim.collision_detector.contacts.model_max_contacts}")
            print(f"[post-step]: sim.collision_detector.contacts.model_num_contacts: {sim.collision_detector.contacts.model_num_contacts}")
            print(f"[post-step]: sim.collision_detector.contacts.world_max_contacts: {sim.collision_detector.contacts.world_max_contacts}")
            print(f"[post-step]: sim.collision_detector.contacts.world_num_contacts: {sim.collision_detector.contacts.world_num_contacts}")
            print(f"[post-step]: sim.collision_detector.contacts.wid: {sim.collision_detector.contacts.wid}")
            print(f"[post-step]: sim.collision_detector.contacts.cid: {sim.collision_detector.contacts.cid}")
            print(f"[post-step]: sim.collision_detector.contacts.body_A:\n{sim.collision_detector.contacts.body_A}")
            print(f"[post-step]: sim.collision_detector.contacts.body_B:\n{sim.collision_detector.contacts.body_B}")
            print(f"[post-step]: sim.collision_detector.contacts.gapfunc:\n{sim.collision_detector.contacts.gapfunc}")
            print(f"[post-step]: sim.collision_detector.contacts.frame:\n{sim.collision_detector.contacts.frame}")
            print(f"[post-step]: sim.collision_detector.contacts.material:\n{sim.collision_detector.contacts.material}")

        # TODO: What to test here?


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=1000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
