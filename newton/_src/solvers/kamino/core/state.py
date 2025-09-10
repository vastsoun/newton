###########################################################################
# KAMINO: Model State Containers
###########################################################################

import warp as wp


class State:
    """
    The compact time-varying state data for a :class:`Model`.
    The compact state data includes rigid body poses and twists, as well as the vector of joint constraint forces.
    The exact attributes depend on the contents of the model.
    State objects should generally be created using the :func:`Model.state()` function.
    """

    def __init__(self):
        self.q_i: wp.array | None = None
        """Array of body coordinates (7-dof transforms) in maximal coordinates with shape ``(nb,)`` and type :class:`transformf`."""
        self.u_i: wp.array | None = None
        """Array of body velocities (6-dof twists) in maximal coordinates with shape ``(nb,)`` and type :class:`vec6f`."""
        self.lambda_j: wp.array | None = None
        """Array of joint constraint forces with shape ``(njd=sum(m_j),)`` and type :class:`float32`."""
