# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: COMPARISON UTILITIES
"""

import unittest
from typing import Any

import numpy as np

from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.utils import logger as msg

###
# Module interface
###

__all__ = [
    "assert_control_equal",
    "assert_state_equal",
]


###
# Utilities
###


def assert_array_attributes_equal(
    test: unittest.TestCase,
    obj0: Any,
    obj1: Any,
    attributes: list[str],
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
    mapping: list[int] | None = None,
    index_remaps: dict[str, list[int]] | None = None,
) -> None:
    """Compare array attributes, permuting rows and remapping referenced indices when requested.

    `mapping` permutes rows of `obj1` to align with `obj0` (e.g. after reordering entities by
    label). `index_remaps` separately translates *values* held by an attribute that reference
    another entity's row in `obj1`'s index space (e.g. a body index) into `obj0`'s index space,
    and applies independently of whether row permutation is active.
    """
    for attr in attributes:
        # Check if attribute exists in both objects
        obj_name = obj0.__class__.__name__
        has_attr0 = hasattr(obj0, attr)
        has_attr1 = hasattr(obj1, attr)
        if not has_attr0 and not has_attr1:
            msg.debug(f"Skipping attribute '{attr}' comparison for {obj_name} because it is missing in both objects.")
            continue
        elif not has_attr0 or not has_attr1:
            test.fail(
                f"Attribute '{attr}' is missing in one of the objects: "
                f" {obj_name} has_attr0={has_attr0}, has_attr1={has_attr1}"
            )
        # Retrieve attributes for logging
        attr0 = getattr(obj0, attr)
        attr1 = getattr(obj1, attr)
        # Check if attributes are array-like
        attr0_is_array = hasattr(attr0, "shape")
        attr1_is_array = hasattr(attr1, "shape")
        if not attr0_is_array and not attr1_is_array:
            msg.debug(
                f"\nSkipping attribute '{obj_name}.{attr}' comparison: both of the objects are not array-like: "
                f"\n0: {obj_name}.{attr}: {type(attr0)}\n1: {obj_name}.{attr}: {type(attr1)}"
            )
            continue
        elif not attr0_is_array or not attr1_is_array:
            test.fail(
                f"Attribute '{attr}' is not array-like in one of the objects: "
                f" {obj_name}.{attr} has_attr0_shape={getattr(attr0, 'shape', None)}, "
                f"has_attr1_shape={getattr(attr1, 'shape', None)}"
            )
        # Test array attribute shapes
        shape0 = attr0.shape
        shape1 = attr1.shape
        test.assertEqual(shape0, shape1, f"{obj_name}.{attr} shapes are not equal.")
        # Test array attribute values
        actual = attr0.numpy()
        desired = attr1.numpy()
        if mapping is not None and len(mapping) == desired.shape[0]:
            desired = desired[mapping]
        if index_remaps is not None and attr in index_remaps:
            desired = np.asarray(desired).copy()
            remap = index_remaps[attr]
            for i, value in enumerate(desired):
                if value >= 0:
                    desired[i] = remap[value]
        # Unbounded limits are stored as inf (e.g. JointsModel.tau_j_max), so this purely
        # informational diff hits inf - inf. Left unguarded it raises a RuntimeWarning, which
        # CI turns into a test error via --strict-warnings.
        with np.errstate(invalid="ignore"):
            diff = actual - desired
        msg.debug("Comparing %s:\nactual:\n%s\ndesired:\n%s\ndiff:\n%s", f"{obj_name}.{attr}", actual, desired, diff)
        np.testing.assert_allclose(
            actual=actual,
            desired=desired,
            err_msg=f"{obj_name}.{attr} are not equal.",
            rtol=rtol.get(attr, 1e-6) if rtol else 1e-6,
            atol=atol.get(attr, 1e-6) if atol else 1e-6,
        )


###
# Container comparisons
###


def assert_state_equal(
    test: unittest.TestCase, state0: StateKamino, state1: StateKamino, excluded: list[str] | None = None
) -> None:
    attributes = [
        "q_i",
        "u_i",
        "w_i",
        "q_j",
        "q_j_p",
        "dq_j",
        "lambda_kin_j",
        "lambda_dyn_j",
        "lambda_f_j",
        "lambda_tau_j",
    ]
    if excluded:
        attributes = [attr for attr in attributes if attr not in excluded]
    assert_array_attributes_equal(test, state0, state1, attributes)


def assert_control_equal(
    test: unittest.TestCase, control0: ControlKamino, control1: ControlKamino, excluded: list[str] | None = None
) -> None:
    attributes = ["tau_j", "q_j_ref", "dq_j_ref", "tau_j_ref"]
    if excluded:
        attributes = [attr for attr in attributes if attr not in excluded]
    assert_array_attributes_equal(test, control0, control1, attributes)
