# Installation

The following instructions detail how to install
the run-time-assurance library on your local system.
It is recommended to install the python modules within
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
run-time-assurance utilizes [Poetry](https://python-poetry.org/) to handle installation.
Poetry can install run-time-assurance into an auto-generated virtualenv or within the currently active environment.

## Installing run-time-assurance

Clone a copy of the run-time-assurance repo onto your local
machine via SSH (recommended):

```shell
git clone git@github.com:act3-ace/run-time-assurance.git
```

or HTTPS:

```shell
git clone https://github.com/act3-ace/run-time-assurance.git
```

### Installing run-time-assurance with pip

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

### Installing run-time-assurance with Poetry

Install the run-time-assurance module into your
environment using `poetry`:

```shell
cd run-time-assurance
poetry install
```

Poetry will handle installing appropriate versions of the dependencies for run-time-assurance into your environment, if they aren't already installed.  Poetry will install an editable version of run-time-assurance to the environment.

## Questions or Issues?

If you have any trouble installing the run-time-assurance
package in your local environment, please feel free to
submit an [issue](https://github.com/act3-ace/run-time-assurance/issues).

For more information on what's available in run-time-assurance,
see our [API](api/index.md).
