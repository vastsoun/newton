# Kamino

**⚠️ Prerelease Software ⚠️**

**This project is in active alpha development.**

Kamino is a solver back-end for the [Newton](https://github.com/newton-physics/newton) physics simulation engine.

Kamino is developed and maintained by [Disney Research](https://www.disneyresearch.com/) in collaboration with [NVIDIA](https://www.nvidia.com/) and [Google DeepMind](https://deepmind.google/).


## Development

Development requires the installation of [Warp](https://github.com/NVIDIA/warp) and [Newton](https://github.com/newton-physics/newton).

First step is to set up a development environment. The simplest is to create a new `virtualenv`. Alternatively one could
follow the [instructions](https://newton-physics.github.io/newton/guide/installation.html) from newton.

### Virtual environments using `pyenv`

Because we're working on a fork of the main Newton repository, it can be usefull to create two `virtualenv|conda|uv` environments.

Using `pyenv` and `virtualenv` it would look something like this:
- one for development of Kamino within our fork
```bash
pyenv virtualenv kamino-dev
pyenv activate kamino-dev
pip install -U pip
```
- and a second for modifying Newton code outside of Kamino that must go through pull requests directly to the main repo:
```bash
pyenv virtualenv newton-dev
pyenv activate newton-dev
pip install -U pip
```

A similar setup can be achieved via `conda|uv`. We've used the `*-dev` suffix to denote environments were the packages will be installed from source, while this can be ommited when creating environments to test installations when installing from wheels.


### Packages

With the target environment enabled, we can proceed to install the necessary packages:

#### APT (Only Required for Linux)
On Linux platforms, e.g. Ubuntu, the following base APT packages must be installed:
```bash
sudo apt-get update
sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
```

#### Warp
Warp can be installed from source (recommended) using:
```bash
git clone git@github.com:NVIDIA/warp.git
cd warp
pip install numpy
python build_lib.py
pip install -e .[dev,extras,docs]
```
Many features and fixes requested by Newton developers come quite often so keeping up to date with Warp `main` can prove usefull.

Alternatively one can install from nightly builds using:
```bash
pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
```

#### Newton
Install Newton from source using:
```bash
git clone git@github.com:vastsoun/newton.git
cd newton
pip install -e .[dev,data,docs]
```

**WARNING**:
Note that currently the development branch where Kamino is being worked on is `dev/kamino`. All PR's within the fork must be issued to that instead of `main` and take care to not to accidentally issue them to `main` of the upstream Newton repository.


## References

The following [technical report](https://arxiv.org/abs/2504.19771) provides a complete summary of the modelling approach and algorithms used within Kamino.

----