# Run Time Assurance

## Intro
The run-time-assurance library provides an interface and implementations for Run Time Assurance (RTA) safety filters. RTA is a technique for guaranteeing the safety of a control system by performing online safety filtering of desired control actions to ensure the control system does not and will not violate defined safety constraints. This allows an unverified controller, such as a neural network, to be utilized in safety critical applications.

This package contains base classes for implementing RTA modules and defining RTA safety constraints. Also included are generic implementations of the RTA algorithms Explicit Simplex, Implicit Simplex, Explicit Acitive Set Invariance Filter (ASIF), and Implicit ASIF which simply require contraints and control system state transition models to become functional RTA modules for any custom application. Additionally, the RTA Zoo includes our growing library of custom RTA implementations for various safety critical control problems.

## Docs
Library documentation and api reference located at [https://rta.git.act3-ace.com/safe-autonomy-stack/run-time-assurance/](https://rta.git.act3-ace.com/safe-autonomy-stack/run-time-assurance/)

## Installation
The following instructions detail how to install 
the run-time-assurance library on your local system.
It is recommended to install the python modules within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

### Installing safe-autonomy-sims
Clone a copy of the safe-autonomy-sims source code 
onto your local machine via SSH:
```shell
git clone git@git.act3-ace.com:rta/run-time-assurance.git
```
or HTTPS:
```shell
git clone https://git.act3-ace.com/rta/run-time-assurance.git
```

Install the run-time-assurance module into your 
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

## Build Docs Locally

First make sure the mkdocs requirements are installed 

```shell
pip install -r mkdocs-requirements.txt
```

Now, build the documentation and serve it locally. By default, you should be able to reach the docs on your local web browser at `127.0.0.1:8000`

```shell
rm -r site
mkdocs build
cp -r docs/. site/
mkdocs serve
```

## Public Release
Distribution A. Approved for public release; distribution is unlimited. Case Number: AFRL-2022-3202

## Team
Umberto Ravaioli,
Kyle Dunlap,
Jamie Cunningham,
John McCarroll,
Kerianne Hobbs
