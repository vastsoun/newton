###########################################################################
# KAMINO: MODELS: MODEL CONSTRUCTION UTILITIES & ASSETS
###########################################################################

from . import builders

__all__ = [
    "builders",
    "get_examples_usd_assets_path",
    "get_primitives_usd_assets_path",
    "get_tests_usd_assets_path",
]

###
# Asset path utilities
###


def get_examples_usd_assets_path() -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/examples/usd")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for example models does not exist: {path}")
    return path


def get_primitives_usd_assets_path() -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/primitives")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for primitive models does not exist: {path}")
    return path


def get_tests_usd_assets_path() -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/tests")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for testing models does not exist: {path}")
    return path
