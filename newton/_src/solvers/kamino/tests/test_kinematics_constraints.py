###########################################################################
# KAMINO: UNIT TESTS: KINEMATICS: CONSTRAINTS
###########################################################################

import unittest
import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.models.builders import build_boxes_fourbar
from newton._src.solvers.kamino.models.utils import (
    make_single_builder,
    make_homogeneous_builder,
    make_heterogeneous_builder,
)

# Test utilities
from newton._src.solvers.kamino.tests.utils.print import (
    print_model_constraint_info,
    print_model_state_info,
)

# Module to be tested
from newton._src.solvers.kamino.kinematics.constraints import make_unilateral_constraints_info


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###

class TestKinematicsConstraints(unittest.TestCase):

    def setUp(self):
        self.verbose = False  # Set to True for detailed output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_single_model_make_constraints(self):
        """
        Tests the population of model info with constraint sizes and offsets for a single-world model.
        """
        # Constants
        max_world_contacts = 20

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)

        # Create the model from the builder
        model: Model = builder.finalize(device=self.default_device)

        # Create a model state
        state = model.data(device=self.default_device)

        # Create a  limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("")
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(capacity=required_world_max_contacts, default_max_contacts=max_world_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            state=state,
            limits=limits,
            contacts=contacts,
            device=self.default_device,
        )
        if self.verbose:
            print(f"model.size:\n{model.size}")
            print_model_constraint_info(model)
            print_model_state_info(state)

    def test_02_homogeneous_model_make_constraints(self):
        """
        Tests the population of model info with constraint sizes and offsets for a homogeneous multi-world model.
        """
        # Constants
        max_world_contacts = 20

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_homogeneous_builder(num_worlds=10, build_func=build_boxes_fourbar)
        num_worlds = builder.num_worlds

        # Create the model from the builder
        model: Model = builder.finalize(device=self.default_device)

        # Create a model state
        state = model.data(device=self.default_device)

        # Create a  limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("")
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(capacity=required_world_max_contacts, default_max_contacts=max_world_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            state=state,
            limits=limits,
            contacts=contacts,
            device=self.default_device,
        )
        if self.verbose:
            print_model_constraint_info(model)
            print_model_state_info(state)

        # Extract numpy arrays from the model info
        model_max_limits = model.size.sum_of_max_limits
        model_max_contacts = model.size.sum_of_max_contacts
        max_limits = model.info.max_limits.numpy()
        max_contacts = model.info.max_contacts.numpy()
        max_limit_cts = model.info.max_limit_cts.numpy()
        max_contact_cts = model.info.max_contact_cts.numpy()
        max_total_cts = model.info.max_total_cts.numpy()
        limits_offset = model.info.limits_offset.numpy()
        contacts_offset = model.info.contacts_offset.numpy()
        unilaterals_offset = model.info.unilaterals_offset.numpy()
        limit_cts_offset = model.info.limit_cts_offset.numpy()
        contact_cts_offset = model.info.contact_cts_offset.numpy()
        unilateral_cts_offset = model.info.unilateral_cts_offset.numpy()
        total_cts_offset = model.info.total_cts_offset.numpy()

        # Check the model info entries
        nj = 0
        njc = 0
        nl = 0
        nlc = 0
        nc = 0
        ncc = 0
        for i in range(num_worlds):
            self.assertEqual(model_max_limits, 4 * num_worlds)
            self.assertEqual(model_max_contacts, max_world_contacts * num_worlds)
            self.assertEqual(max_limits[i], 4)
            self.assertEqual(max_contacts[i], max_world_contacts)
            self.assertEqual(max_limit_cts[i], 4)
            self.assertEqual(max_contact_cts[i], 3 * max_world_contacts)
            self.assertEqual(max_total_cts[i], 20 + 4 + 3 * max_world_contacts)
            self.assertEqual(limits_offset[i], nl)
            self.assertEqual(contacts_offset[i], nc)
            self.assertEqual(unilaterals_offset[i], nl + nc)
            self.assertEqual(limit_cts_offset[i], nlc)
            self.assertEqual(contact_cts_offset[i], ncc)
            self.assertEqual(unilateral_cts_offset[i], nlc + ncc)
            self.assertEqual(total_cts_offset[i], njc + nlc + ncc)
            nj += 4
            njc += 20
            nl += 4
            nlc += 4
            nc += max_world_contacts
            ncc += 3 * max_world_contacts

    def test_03_heterogeneous_model_make_constraints(self):
        """
        Tests the population of model info with constraint sizes and offsets for a heterogeneous multi-world model.
        """
        # Constants
        max_world_contacts = 20

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_heterogeneous_builder()

        # Create the model from the builder
        model: Model = builder.finalize(device=self.default_device)

        # Create a model state
        state = model.data(device=self.default_device)

        # Create a  limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("")
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(capacity=required_world_max_contacts, default_max_contacts=max_world_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            state=state,
            limits=limits,
            contacts=contacts,
            device=self.default_device,
        )
        if self.verbose:
            print_model_constraint_info(model)
            print_model_state_info(state)
            print("state.info.num_limits.ptr: ", state.info.num_limits.ptr)
            print("limits.world_num_limits.ptr: ", limits.world_num_limits.ptr)
            print("state.info.num_contacts.ptr: ", state.info.num_contacts.ptr)
            print("contacts.world_num_contacts.ptr: ", contacts.world_num_contacts.ptr)

        # Check if the state info entity counters point to the same arrays as the limits and contacts containers
        self.assertTrue(state.info.num_limits.ptr, limits.world_num_limits.ptr)
        self.assertTrue(state.info.num_contacts.ptr, contacts.world_num_contacts.ptr)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
