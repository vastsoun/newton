###########################################################################
# KAMINO: Model Control Containers
###########################################################################

import warp as wp


class Control:
    """
    The compact time-varying control data for a :class:`Model`.
    The compact control data includes controllable generalized forces of the joints.
    The exact attributes depend on the contents of the model.
    Control objects should generally be created using the :func:`Model.control()` function.
    """
    def __init__(self):
        self.tau_j: wp.array | None = None
        """Array of joint control forces with shape ``(sum(nqd_w),)`` and type :class:`float`."""
