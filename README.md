# Run Time Assurance

## Intro
The run-time-assurance library provides an interface and implementations for Run Time Assurance (RTA) safety filters. RTA is a technique for guaranteeing the safety of a control system by performing online safety filtering of desired control actions to ensure the control system does not and will not violate defined safety constraints. This allows an unverified controller, such as a neural network, to be utilized in safety critical applications.

This package contains base classes for implementing RTA modules and defining RTA safety constraints. Also included are generic implementations of the RTA algorithms Explicit Simplex, Implicit Simplex, Explicit Acitive Set Invariance Filter (ASIF), and Implicit ASIF which simply require contraints and control system state transition models to become functional RTA modules for any custom application. Additionally, the RTA Zoo includes our growing library of custom RTA implementations for various safety critical control problems.


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
git clone git@github.com/act3-ace:rta/run-time-assurance.git
```
or HTTPS:
```shell
git clone https://github.com/act3-ace/run-time-assurance.git
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

## Team
Umberto Ravaioli,
Kyle Dunlap,
Jamie Cunningham,
John McCarroll,
Kerianne Hobbs
