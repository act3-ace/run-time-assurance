#########################################################################################
# develop stage contains base requirements. Used as base for all other stages
#########################################################################################
ARG ACT3_OCI_REGISTRY=reg.git.act3-ace.com
ARG IMAGE_REPO_BASE
ARG SA_DYNAMICS_TAG=0.0.1

FROM ${ACT3_OCI_REGISTRY}/rta/safe-autonomy-stack/safe-autonomy-dynamics/releases/package:v0.0.1001 as sa_dynamics_package

FROM ${IMAGE_REPO_BASE}docker.io/python:3.8 as develop

ARG CI_JOB_TOKEN

ARG PIP_INDEX_URL
ARG APT_MIRROR_URL
ARG SECURITY_MIRROR_URL
ARG SA_DYNAMICS_TAG

#Sets up apt mirrors to replace the default registries
RUN if [ -n "$APT_MIRROR_URL" ] ; then sed -i "s|http://archive.ubuntu.com|${APT_MIRROR_URL}|g" /etc/apt/sources.list ; fi && \
if [ -n "$SECURITY_MIRROR_URL" ] ; then sed -i "s|http://security.ubuntu.com|${SECURITY_MIRROR_URL}|g" /etc/apt/sources.list ; fi

# Clone dynamics repo
COPY --from=sa_dynamics_package /opt/libact3-sa-dynamics /opt/libact3-sa-dynamics

RUN python --version && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        /opt/libact3-sa-dynamics/safe_autonomy_dynamics-{SA_DYNAMICS_TAG}-py3-none-any.whl 

#########################################################################################
# build stage packages the source code
#########################################################################################

FROM develop as build
ENV RTA_ROOT=/opt/libact3-rta

WORKDIR /opt/project
COPY . .

RUN python setup.py bdist_wheel -d ${RTA_ROOT} && \
    pip install --no-cache-dir .

#########################################################################################
# package stage 
#########################################################################################

# the package stage contains everything required to install the project from another container build
# NOTE: a kaniko issue prevents the source location from using a ENV variable. must hard code path

FROM scratch as package
COPY --from=build /opt/libact3-rta /opt/libact3-rta

#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.
FROM build as cicd
