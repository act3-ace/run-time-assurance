# [1.16.0](https://git.act3-ace.com/rta/safe-autonomy-stack/run-time-assurance/compare/v1.15.3...v1.16.0) (2024-3-4)


### Features

* Dynamically changing parameters ([5791dad](https://git.act3-ace.com/rta/safe-autonomy-stack/run-time-assurance/commit/5791dad03561853f90a36458c2719224a1e2c9b0))

## 1.15.3 (2023-09-21)


### Bug Fixes

* **requirements:** update required sa-dynamics version to 0.13.2 96254f6

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

## [1.10.1](https://github.com/act3-ace/run-time-assurance/compare/v1.10.0...v1.10.1) (2023-03-28)


### Bug Fixes

* use jax 0.4.3 ([305f017](https://github.com/act3-ace/run-time-assurance/commit/305f0171fc020bab523f22036031451a743e9d34))

# [1.10.0](https://github.com/act3-ace/run-time-assurance/compare/v1.9.2...v1.10.0) (2023-03-16)


### Features

* Created SA-Dynamics ASIF ode solver with support for jax ode integration ([1b45175](https://github.com/act3-ace/run-time-assurance/commit/1b45175f5cf3cf4277e08bda8e713d03f75ce432))

## [1.9.2](https://github.com/act3-ace/run-time-assurance/compare/v1.9.1...v1.9.2) (2023-03-15)


### Bug Fixes

* Refactor tests to use DataTrackingSampleTestingModule ([4f848bc](https://github.com/act3-ace/run-time-assurance/commit/4f848bcc2374f1db6c033836a4ba0e113fb256c0))

## [1.9.1](https://github.com/act3-ace/run-time-assurance/compare/v1.9.0...v1.9.1) (2023-03-01)


### Bug Fixes

* Use jnp array instead of list for PSM ([f60d869](https://github.com/act3-ace/run-time-assurance/commit/f60d869a0c52ab1f0e2b6ede91457fed0429556c))

# [1.9.0](https://github.com/act3-ace/run-time-assurance/compare/v1.8.0...v1.9.0) (2023-01-25)


### Features

* **constraint:** Created inequality constraint class that can be added to QP ([e60fb7f](https://github.com/act3-ace/run-time-assurance/commit/e60fb7fefe97cb00499f1720aa47fe48dc83c38e))

# [1.8.0](https://github.com/act3-ace/run-time-assurance/compare/v1.7.0...v1.8.0) (2023-01-24)


### Features

* **constraint:** Resolve "Incorporate exponential/high order CBFs" ([66e53aa](https://github.com/act3-ace/run-time-assurance/commit/66e53aa78c7b7671fe017ae56b8e026e09a58b11))
* **jit:** Disable jit and vmap for debugging ([45359db](https://github.com/act3-ace/run-time-assurance/commit/45359dbe97010a5eca24d30e58219a8050bfa882))

# [1.7.0](https://github.com/act3-ace/run-time-assurance/compare/v1.6.0...v1.7.0) (2023-01-24)


### Features

* **zoo:** 1d Integrator RTA ([b6448c9](https://github.com/act3-ace/run-time-assurance/commit/b6448c987e31c6b5d33c4c41d6077a346896a3b5))

# [1.6.0](https://github.com/act3-ace/run-time-assurance/compare/v1.5.0...v1.6.0) (2023-01-18)


### Features

* **logging:** constraint values are logged by rta modules ([0a253b3](https://github.com/act3-ace/run-time-assurance/commit/0a253b360bb6f107c58c20a099e523329afc5e29))

# [1.5.0](https://github.com/act3-ace/run-time-assurance/compare/v1.4.0...v1.5.0) (2023-01-16)


### Features

* **implicit asif:** Fix subsample_constraints for implicit ASIF ([4f6a7f4](https://github.com/act3-ace/run-time-assurance/commit/4f6a7f4cc2ddee25c6c9c81c90efd014a47010b3))

# [1.4.0](https://github.com/act3-ace/run-time-assurance/compare/v1.3.0...v1.4.0) (2023-01-16)


### Features

* **dynamics:** Default ASIF predicted state method ([e052c7c](https://github.com/act3-ace/run-time-assurance/commit/e052c7c403f8dcbd22a9ffc7d0008b0af2c35339))

# [1.3.0](https://github.com/act3-ace/run-time-assurance/compare/v1.2.0...v1.3.0) (2022-11-14)


### Features

* **zoo:** Inspection RTA ([d5c489f](https://github.com/act3-ace/run-time-assurance/commit/d5c489f809b99e619537469f57253c0a75359003))

# [1.2.0](https://github.com/act3-ace/run-time-assurance/compare/v1.1.1...v1.2.0) (2022-11-01)


### Features

* **constraint:** constraint bias and monte carlo compat ([941f7e5](https://github.com/act3-ace/run-time-assurance/commit/941f7e56e8feb8b1ea17f52e2ef5ed29e5b3ad03))

## [1.1.1](https://github.com/act3-ace/run-time-assurance/compare/v1.1.0...v1.1.1) (2022-08-18)


### Bug Fixes

* **setup.py:** fix versioning ([89ead8e](https://github.com/act3-ace/run-time-assurance/commit/89ead8e99ea28f0c1b8932e6d64f457185553cee))

# [1.1.0](https://github.com/act3-ace/run-time-assurance/compare/v1.0.0...v1.1.0) (2022-08-18)


### Bug Fixes

* add version stuff for package ([a74ffcf](https://github.com/act3-ace/run-time-assurance/commit/a74ffcf565518644499c55930ec2fa5b47e0a5c5))
* **Dockerfiel:** typo ([5616740](https://github.com/act3-ace/run-time-assurance/commit/5616740fed9cd888c1ff2778597fc8dc8cace092))
* **Dockerfile:** add temp version for build to pass ([8ffe626](https://github.com/act3-ace/run-time-assurance/commit/8ffe626369643a92bd1394d57ab66cae0e1c5a3b))
* **Dockerfile:** explicitly name version ([ed580c7](https://github.com/act3-ace/run-time-assurance/commit/ed580c708d575bbcd6c32a1ff829c9133f1fb4d8))
* **Dockerfile:** minor typo fixes ([821983a](https://github.com/act3-ace/run-time-assurance/commit/821983a2919312292deb28be0ecfd66531628548))
* **Dockerfile:** replace content ([6abedd2](https://github.com/act3-ace/run-time-assurance/commit/6abedd2ec65ecd8a2698fa95269b1743a8ba6dab))
* **Dockerfile:** stupid computer ([9736db0](https://github.com/act3-ace/run-time-assurance/commit/9736db0167ac8cf5aadfbe8bc9fc7f9b92ce7615))
* **Dockerfile:** try again with arg ([4caa7be](https://github.com/act3-ace/run-time-assurance/commit/4caa7be727b6080e523a1ab20fd2313be4523946))
* **Dockerfile:** typo ([b03bbb3](https://github.com/act3-ace/run-time-assurance/commit/b03bbb3f7391b7cb442f068363730c5691688715))
* **dockerfile:** updat version ([5267338](https://github.com/act3-ace/run-time-assurance/commit/5267338eb3d471db5bc66c75d387798eac1c8cd4))
* **Dockerfile:** update for package dependencies, remove git clone ([334c3f3](https://github.com/act3-ace/run-time-assurance/commit/334c3f39ec02b9dd2b120b826d480b93cabc6ded))
* **Dockerfile:** update for sa-dynamics changes ([7b27c2d](https://github.com/act3-ace/run-time-assurance/commit/7b27c2d9c6ba484669150e976d9672be0f8fd165))
* **docker:** update sa dynamics to 0.3.0 ([c01ee32](https://github.com/act3-ace/run-time-assurance/commit/c01ee3294ded7be264162d5a03f0926e19bfb76f))
* **gitlab-ci:** image dep for mkkdocs ([7c77d83](https://github.com/act3-ace/run-time-assurance/commit/7c77d8335060ec8f315afa7c0328839c3b27ed9a))
* **gitlab-ci:** no allow fail on mkdocs ([159f925](https://github.com/act3-ace/run-time-assurance/commit/159f92567020650a46423edd3685e4a99597d4ae))
* **gitlab-ci:** update cicd ([f1fc16b](https://github.com/act3-ace/run-time-assurance/commit/f1fc16b374100e7446c8e3fa6a628294da72958c))
* **gitlab-ci:** update mkdocs ([cc3a506](https://github.com/act3-ace/run-time-assurance/commit/cc3a506084ed35fc15a72789a02b9b497d575496))
* **gitlab-ci:** update mkdocs image ([52e9bc0](https://github.com/act3-ace/run-time-assurance/commit/52e9bc0f337ff94bb6c0abb0103f8df6cbd9c676))
* image path ([0934e18](https://github.com/act3-ace/run-time-assurance/commit/0934e18c520111ea3d8a25778b10c90c8a69eb9a))
* more mkdocds fix ([a119ad0](https://github.com/act3-ace/run-time-assurance/commit/a119ad01bad7028a580dd746836edc451dd6bd53))
* pin mkdocsstrings ([14cdd73](https://github.com/act3-ace/run-time-assurance/commit/14cdd73eed46fa1631c3f2c233dc0c59328aa3eb))
* **pylint:** fix pylint errors ([e16d75d](https://github.com/act3-ace/run-time-assurance/commit/e16d75d77c93e1f364da1659fdb75bbcfd0f8ba9))
* remove version file and lint disable ([46d10e8](https://github.com/act3-ace/run-time-assurance/commit/46d10e85660bdc909a00d23b64a17812a82d0299))
* semantic release files ([37cc3cf](https://github.com/act3-ace/run-time-assurance/commit/37cc3cf37c988cfb25b5b075e4e02bfe2c8530bf))
* try old mkdocs ([4bde0ab](https://github.com/act3-ace/run-time-assurance/commit/4bde0ab9fd4e6dac613212c039cbd349f9746c10))
* updat mkdocs to allow failure ([b4c48fb](https://github.com/act3-ace/run-time-assurance/commit/b4c48fb6cf1104b684d2f583cc2237a044b9cfdd))
* update semantic release items ([bd02af1](https://github.com/act3-ace/run-time-assurance/commit/bd02af1237cc2c2c59add29e919aa0d6823ae513))


### Features

* **Dockerfile:** update Dockerfile verrsion ([cb5cc93](https://github.com/act3-ace/run-time-assurance/commit/cb5cc93374a43c79ea5dcf054217b6faf0231b7b))
