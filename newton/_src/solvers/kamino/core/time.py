###########################################################################
# KAMINO: Time Module
###########################################################################

from __future__ import annotations

import warp as wp

from .types import int32, float32


###
# Module interface
###

__all__ = [
    "TimeModel",
    "TimeData",
    "advance_time",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers Containers
###


class TimeModel:
    """
    A container to hold the time-invariant gravity model data.
    """
    def __init__(self):
        self.dt: wp.array(dtype=float32) | None = None
        """
        The discrete time-step size of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

        self.inv_dt: wp.array(dtype=float32) | None = None
        """
        The inverse of the discrete time-step size of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

    def set_timestep(self, dt: float):
        """
        Set the time-step size for each world.

        Args:
            dt (float): The time-step size to set.
        """
        self.dt.fill_(dt)
        self.inv_dt.fill_(1.0 / dt)


class TimeData:
    """
    A container to hold the time-invariant gravity model data.
    """
    def __init__(self):
        self.steps: wp.array(dtype=int32) | None = None
        """
        The current number of simulation steps of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.time: wp.array(dtype=float32) | None = None
        """
        The current simulation time of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

    def zero(self):
        """
        Reset the time state to zero.
        """
        self.steps.fill_(0)
        self.time.fill_(0.0)


###
# Kernels
###

@wp.kernel
def _advance_time(
    # Inputs
    dt: wp.array(dtype=float32),
    # Outputs
    steps: wp.array(dtype=int32),
    time: wp.array(dtype=float32),
):
    """
    Reset the current state to the initial state defined in the model.
    """
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Update the time and step count
    steps[wid] += 1
    time[wid] += dt[wid]


###
# Launchers
###

def advance_time(model: TimeModel, data: TimeData):

    # Ensure the model is valid
    if model is None:
        raise ValueError("'model' must be initialized, is None.")
    elif not isinstance(model, TimeModel):
        raise TypeError("'model' must be an instance of TimeModel.")
    if model.dt is None:
        raise ValueError("'model' must has a `model.dt` array, is None.")

    # Ensure the state is valid
    if data.steps is None:
        raise ValueError("'data' must has a `data.steps` array, is None.")
    elif not isinstance(data, TimeData):
        raise TypeError("'data' must be an instance of TimeData.")
    if data.time is None:
        raise ValueError("'data' must has a `data.time` array, is None.")

    # Retrieve the number of worlds from an array size
    # NOTE: It is assumed that all arrays are of the same size.
    num_worlds = model.dt.size

    #
    wp.launch(
        _advance_time,
        dim=num_worlds,
        inputs=[
            # Inputs:
            model.dt,
            # Outputs:
            data.steps,
            data.time
        ]
    )