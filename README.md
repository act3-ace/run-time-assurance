# Run Time Assurance

## Intro

The run-time-assurance library provides an interface and implementations for Run Time Assurance (RTA) safety filters. RTA is a technique for guaranteeing the safety of a control system by performing online safety filtering of desired control actions to ensure the control system does not and will not violate defined safety constraints. This allows an unverified controller, such as a neural network, to be utilized in safety critical applications.

This package contains base classes for implementing RTA modules and defining RTA safety constraints. Also included are generic implementations of the RTA algorithms Explicit Simplex, Implicit Simplex, Explicit Acitive Set Invariance Filter (ASIF), and Implicit ASIF which simply require contraints and control system state transition models to become functional RTA modules for any custom application. Additionally, the RTA Zoo includes our growing library of custom RTA implementations for various safety critical control problems.

## Docs

Library documentation and api reference located at [https://rta.github.com/act3-ace/safe-autonomy-stack/run-time-assurance/](https://rta.github.com/act3-ace/safe-autonomy-stack/run-time-assurance/)

## Installation

The following instructions detail how to install
the run-time-assurance library on your local system.
It is recommended to install the python modules within
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
run-time-assurance utilizes [Poetry](https://python-poetry.org/) to handle installation.
Poetry can install run-time-assurance into an auto-generated virtualenv or within the currently active environment.

### Installing run-time-assurance

Clone a copy of the run-time-assurance source code
onto your local machine via SSH:

```shell
git clone git@github.com:act3-ace/run-time-assurance.git
```

or HTTPS:

```shell
git clone git@github.com:act3-ace/run-time-assurance.git
```

You can install the run-time-assurance module using either `pip` or `poetry`.

#### Installing run-time-assurance with pip

To install the run-time-assurance module into your
environment using `pip`:

```shell
cd run-time-assurance
pip install .
```

For a local development version, please install
using the `-e, --editable` option:

```shell
pip install -e .
```

#### Installing run-time-assurance with Poetry

To install the run-time-assurance module into your
environment using `poetry`:

```shell
cd run-time-assurance
poetry install
```

Poetry will handle installing appropriate versions of the dependencies for run-time-assurance into your environment, if they aren't already installed.  Poetry will install an editable version of run-time-assurance to the environment.

## Build Docs Locally

First make sure the mkdocs requirements are installed

```shell
poetry install --with docs
```

Now, build the documentation and serve it locally. By default, you should be able to reach the docs on your local web browser at `127.0.0.1:8000`

```shell
rm -r site
poetry run mkdocs build
cp -r docs/. site/
poetry run mkdocs serve
```

## Tutorial

A Jupyter Notebook tutorial discussing how to build RTA modules is given at `run-time-assurance/tutorials/Double_Integrator_Tutorial.ipynb`.

## Citation

The [paper](https://arxiv.org/pdf/2209.01120.pdf) associated with this repository was presented at the 2023 American Control Conference, and can be cited with the following bibtex entry:

```tex
@inproceedings{ravaioli2023universal,
  title={A Universal Framework for Generalized Run Time Assurance with JAX Automatic Differentiation},
  author={Ravaioli, Umberto J and Dunlap, Kyle and Hobbs, Kerianne},
  booktitle={2023 American Control Conference (ACC)},
  pages={4264--4269},
  year={2023},
  organization={IEEE}
}
```

## Public Release

Approved for public release; distribution is unlimited. Case Number: AFRL-2023-6154

A prior version of this repository was approved for public release. Case Number: AFRL-2022-3202

## Team

Umberto Ravaioli,
Kyle Dunlap,
Jamie Cunningham,
John McCarroll,
Kerianne Hobbs,
Charles Keating
