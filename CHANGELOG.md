## [1.18.5](https://github.com/act3-ace/run-time-assurance/compare/v1.18.4...v1.18.5) (2024-10-25)


### Bug Fixes

* **jax:** don't jit compile external function calls ([6f591c3](https://github.com/act3-ace/run-time-assurance/commit/6f591c355c676605b7e99d6f99d19f41734167f9)), closes [#12](https://github.com/act3-ace/run-time-assurance/issues/12)

## [1.18.4](https://github.com/act3-ace/run-time-assurance/compare/v1.18.3...v1.18.4) (2024-07-31)


### Bug Fixes

* **jax:** remove explicit jax flags for sim api ([0fa1444](https://github.com/act3-ace/run-time-assurance/commit/0fa144400f03c229242e184e89c19fbf337c79d8))

## 1.18.3 (2024-07-11)


### Bug Fixes

* **safe-autonomy-simulation:** upgrade to safe-autonomy-simulation v2 dynamics

## ## 1.18.2 (2024-06-27)


### Bug Fixes

* integrator tutorial updates

## 1.18.1 (2024-06-26)


### Bug Fixes

* add params to integrator constraints

# 1.18.0 (2024-06-24)


### Features

* Track constraints that cause intervention

# 1.17.0 (2024-04-15)


### Features

* **sim:** change sim backend to safe-autonomy-simulation

## [1.16.1](https://github.com/act3-ace/run-time-assurance/compare/v1.16.0...v1.16.1) (2024-04-05)


### Bug Fixes

* bump jax verstion to 0.4.26 ([7e9688a](https://github.com/act3-ace/run-time-assurance/commit/7e9688a671cfdd5d445dffbbd41610f2fabced26))
* **dependencies:** updated safe-autonomy-dynamics to 1.2.3 with jax extra ([b4c077d](https://github.com/act3-ace/run-time-assurance/commit/b4c077d04417a65d29af6cb7efdc0df0956410c3))

# [1.16.0](https://github.com/act3-ace/run-time-assurance/compare/v1.15.3...v1.16.0) (2024-03-26)


### Features

* Dynamically changing parameters ([2d4aa95](https://github.com/act3-ace/run-time-assurance/commit/2d4aa95e818b0d901632882aec40bff017526c6d))

## [1.15.3](https://github.com/act3-ace/run-time-assurance/compare/v1.15.2...v1.15.3) (2024-01-10)


### Bug Fixes

* **requirements:** update required sa-dynamics version to 0.13.2 ([2842994](https://github.com/act3-ace/run-time-assurance/commit/284299422aac33c222291b25242e281fa701b3be))

## 1.15.2 (2023-09-20)


### Bug Fixes

* **requirements:** update required sa-dynamics (0.12.0 -> 0.13.1) and python version (3.8 -> 3.10.5) f23e4f1

## 1.15.1 (2023-07-05)


### Bug Fixes

* slack priority for HOCBFs b813846

# 1.15.0 (2023-06-20)


### Features

* Slack variables for ASIF optimization objective b052e3d

## 1.14.1 (2023-06-20)


### Bug Fixes

* Update to safe-autonomy-dynamics 0.12.0 b02ac40

# 1.14.0 (2023-06-14)


### Features

* Discrete cbfs for inspection 26fbb84

# 1.13.0 (2023-06-14)


### Features

* Adjust subsampling logic 7bf7a27

# 1.12.0 (2023-06-12)


### Features

* Discrete CBF module c6b5f8e

## 1.11.1 (2023-06-09)


### Bug Fixes

* copy constraint dict before modifying ede6e34

# 1.11.0 (2023-06-09)


### Features

* Filter constraints based on list of strings 9d8388a

## 1.10.7 (2023-04-26)


### Bug Fixes

* Inspection Constraint Sqrt Gradient Issues 98eda71
* Logging updates aabf244

## 1.10.6 (2023-04-13)


### Bug Fixes

* update dependency to dynamics 0.11.5 c099caf

## 1.10.5 (2023-04-12)


### Bug Fixes

* added build args to build tagged image and package image 5612250

## 1.10.4 (2023-04-12)


### Bug Fixes

* version test 8c1bed4

## 1.10.2 (2023-04-07)


### Bug Fixes

* removed semantic release commit message repo links c37d7b6

## 1.10.1 (2023-03-28)


### Bug Fixes

* use jax 0.4.3

# 1.10.0 (2023-03-16)


### Features

* Created SA-Dynamics ASIF ode solver with support for jax ode integration

## 1.9.2 (2023-03-15)


### Bug Fixes

* Refactor tests to use DataTrackingSampleTestingModule

## 1.9.1 (2023-03-01)


### Bug Fixes

* Use jnp array instead of list for PSM

# 1.9.0 (2023-01-25)


### Features

* **constraint:** Created inequality constraint class that can be added to QP

# 1.8.0 (2023-01-24)


### Features

* **constraint:** Resolve "Incorporate exponential/high order CBFs"
* **jit:** Disable jit and vmap for debugging

# 1.7.0 (2023-01-24)


### Features

* **zoo:** 1d Integrator RTA

# 1.6.0 (2023-01-18)


### Features

* **logging:** constraint values are logged by rta modules

# 1.5.0 (2023-01-16)


### Features

* **implicit asif:** Fix subsample_constraints for implicit ASIF

# 1.4.0 (2023-01-16)


### Features

* **dynamics:** Default ASIF predicted state method

# 1.3.0 (2022-11-14)


### Features

* **zoo:** Inspection RTA

# 1.2.0 (2022-11-01)


### Features

* **constraint:** constraint bias and monte carlo compat

## 1.1.1 (2022-08-18)


### Bug Fixes

* **setup.py:** fix versioning

# 1.1.0 (2022-08-18)


### Bug Fixes

* add version stuff for package
* **Dockerfiel:** typo
* **Dockerfile:** add temp version for build to pass
* **Dockerfile:** explicitly name version
* **Dockerfile:** minor typo fixes
* **Dockerfile:** replace content
* **Dockerfile:** stupid computer
* **Dockerfile:** try again with arg
* **Dockerfile:** typo
* **dockerfile:** updat version
* **Dockerfile:** update for package dependencies, remove git clone
* **Dockerfile:** update for sa-dynamics changes
* **docker:** update sa dynamics to 0.3.0
* **gitlab-ci:** image dep for mkkdocs
* **gitlab-ci:** no allow fail on mkdocs
* **gitlab-ci:** update cicd
* **gitlab-ci:** update mkdocs
* **gitlab-ci:** update mkdocs image
* image path
* more mkdocds fix
* pin mkdocsstrings
* **pylint:** fix pylint errors
* remove version file and lint disable
* semantic release files
* try old mkdocs
* updat mkdocs to allow failure
* update semantic release items


### Features

* **Dockerfile:** update Dockerfile verrsion
