# Developer Guide

## Design Patterns

run-time-assurance is designed around the concept of a [```RTAModule```](api/rta/base.md#run_time_assurance.rta.base.RTAModule), which provides the functionaltiy common to all RTA methods.

## Testing

run-time-assurance currently does not have unit tests to verify the functionality of the library.  However, there is sample testing available that run large Monte Carlo simulations to determine what initial conditions will cause RTA to fail.  While these tests may not be useful for verifying functionality as part of a CI/CD pipeline, these tests can be useful as part of development of RTA methods.  You can run a sample test for 2D docking by running

```shell
python -m run_time_assurance.zoo.cwh.random_sample_testing
```

The test will generate results in a __tmp__ folder with results in a CSV showing whether RTA ensured safety for a given initial condition.

## Releasing

Releases of run-time-assurance are handled by the CI/CD pipeline.  New releases of run-time-assurance will be created with each update to the main branch of the repository.  The CI/CD pipeline will handle verifying the state of the code, and upon successful verification and testing, will publish a package of run-time-assurance to an appropriate package repository, such as PyPI.
