# Setup and installation of software

## On your own PC

### Installing a package and environment manager

Download and install [(mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Update conda
```bash
conda update conda
```

Add `conda-forge` channel (allows installation of more up to date packages)
```bash
conda config --add channels conda-forge
```

### Clone the git repo

```bash
git clone https://github.com/leyffer/m2dtIceSheet.git
```

### Create virtual environment and install python packages

To create a virtual environment with the name `fenics-env` and install the package dependencies in `requirements.txt` (in the git repo):
```bash
conda create -n fenics-env --file requirements.txt
```

This might not work exactly as each computer has different hardware (versions may not be compatible across machines).

Creating the environment without installing all of the packages:
```bash
conda create -n fenics-env
```

> Instead of `requirements.txt`, we can also try using `environments/mtdt_oed_FEniCS.yml`
> ```bash
> conda create -n fenics-env --file environments/mtdt_oed_FEniCS.yml
> ```

#### Dependencies

Environment files (e.g., `requirements.txt`) are a nice convenience that indicates a working set of dependencies and versions.

If you run into problems with those versions and packages, the software dependencies for this project are:
```
numpy
scipy
fenics-dolfin
mpich
pyvista
cyipopt
```

Most of the code is set up to run in Jupyter notebooks.
```bash
conda install -c conda-forge jupyterlab
```

#### FEniCS and FEniCSx

We are using legacy software FEniCS for the finite element problems here. This is compatible with hIPPYlib, but we have not added hIPPYlib support (yet).

FEniCSx is the updated version of FEniCS. Firedrake is another alternative, but is a headache to install. Each different software has different conventions that make translating code from one to the other a headache.

The [webpage for installing FEniCSx](https://fenicsproject.org/download/) indicates that this is easiest when done using `apt` on linux machines (below edited for fenics instead of fenicsx):
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenics
```

> NOTE: installing FEniCS takes a while. It will build PETSc and some other tools which will take a long time. Installing Firedrake is even worse although I was able to get Firedrake working on my Mac and FEniCS refused to work.

### Adding packages

```bash
conda install <package-name>
```

If, for some reason, the package is not available using conda, you can also use `pip`, but this does not have the same guarantees that the installed packages will be compatable with one another:
```bash
pip install <package-name>
```

### Activating the virtual environment

```bash
conda activate fenics-env
```

## On the GCE

### Activating the virtual environment

> You may need to load the anaconda3 module in the GCE environment first.
> ```bash
> module avail
> module load anaconda3
> ```

An existing environment exists on the GCE and can be activated using:
```bash
conda activate /nfs/gce/projects/M2dt-OED/fenics-env/
```
This will require the least effort to get working, just requires establishing a connection to the GCE system.

## Repo structure

The code is organized as follows:

- `source` contains the main source classes and code.
    - Note that the `State` class contains, within itself, an attribute called `state` that is a model PDE solution, i.e., you must, confusingly, provide a `state` to the `State` constructor
- `source/Optimization` contains the code used for optimizing. At the moment, the only code that is really fleshed out is the `DAE.py` file containing an implementation of a discretized DAE using cyipopt (python implementation of IPOPT).
- `models` contains different possible PDE models (right now only the advection diffusion equation)
- `models/AdvectionDiffusion` contains model specific class implementations
- `models/AdvectionDiffusion/Detectors` contains various classes for measuring a state
    - `DetectorApprox` smoothly approximates the provided state (for steady-state states) and emulates the other `Detector` classes
- `models/AdvectionDiffusion/Navigators` contains various classes for transforming path design variables into a path (and provide derivatives)
    - The only currently used class for the DAE optimization from here is the `NavigationFreePath.py` which takes the complete path (discretized) as the design variables (there are no derivatives with respect to design variables)
    - The confusingly named `CircularPath` and `CirclePath` are similar (apologies for the confusion; the navigator classes need to be cleaned up). The `CirclePath` is a specific instance of a `CircularPath` with a fixed center and specified by a radius and velocity. The `CircularPath` is more general and is specified by velocity, angular velocity, initial heading, and initial position.
    - For optimization, all of these different classes can simply be gotten rid of and appropriate constraints added. For analysis of the system, the `CirclePath` remains useful as it is a simplifies two-control system that can be easily analyzed.
- `models/AdvectionDiffusion/settings` contains different implementations of the advection diffusion problem (FEniCS and Firedrake). However, we are not using Firedrake anymore and that code should probably be removed
- `models/AdvectionDiffusion/settings/AdvectionDiffusion_FEniCS` contains problem specific implementations of the `Drone`, `State` and full order model `FOM` and `FOM_stationary`
- `models/AdvectionDiffusion/settings/EdgeDrone` contains some code related to the graph based drone implementation that we had previously discussed but did not fully flesh out. Most of this code is old and out of date. The `GraphEdges.py`, `MyEdgeDrone.py` and `MyGraphDrone.py` are the only relevant code files, the others should probably be removed. This code should also just be moved into the `Navigator` directory.

I tried to make importing code from these various directories less painful by adding `__init__.py` files all over, but this did not work. This means that importing code from disparate directories requires adding those individual directories to the PATH variable, something that makes development more confusing and importing (at all and specific code) more challenging.

## Code Structure

A `FullOrderModel`object holds the PDE model and can compute FEM solutions to the PDE provided some parameters (e.g., a basis forcing term to create a basis PDE response for a linear PDE). The prior is defined in this object.

A `State` object holds the PDE state and a reference to the parameters/basis that define it.

A `Detector` returns measurements from a PDE state.

A `Navigator` contains logic for transforming control parameters into paths. 

A `Flight` contains a flightpath and a time grid as well as a reference to the parameters that specified that path.

A `Drone` Object holds a `Detector` and a `Navigator` and plans `Flight` objects using the `Navigator`.

A `Posterior` object computes posterior mean and covariance.

An `InverseProblem` references the `FullOrderModel` and requests solutions to basis forcing terms to construct the PDE response basis. The `InverseProblem` uses these to request data from each PDE response from the referenced `Drone` and then uses these data to compute a `Posterior` object.

An `OEDUtility` computes an OED utility function (and derivatives) given a `Posterior` covariance.
