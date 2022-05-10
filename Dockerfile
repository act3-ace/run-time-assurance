##########################################################################################
# Dependent tags
##########################################################################################

ARG CORL_TAG=1.42.0
ARG ACT3_OCI_REGISTRY=reg.git.act3-ace.com

##########################################################################################
# Dependent images
##########################################################################################

FROM ${ACT3_OCI_REGISTRY}/act3-rl/corl/releases/package:v${CORL_TAG} as corl_package

#########################################################################################
# develop stage contains base requirements. Used as base for all other stages
#########################################################################################

ARG IMAGE_REPO_BASE
FROM ${IMAGE_REPO_BASE}docker.io/python:3.8 as develop

ARG PIP_INDEX_URL
ARG CORL_TAG

#Sets up apt mirrors to replace the default registries
RUN echo "deb ${APT_MIRROR_URL} stable main contrib non-free" > /etc/apt/sources.list && \
echo "deb-src ${APT_MIRROR_URL} stable main contrib non-free" >> /etc/apt/sources.list

# copy in corl
COPY --from=corl_package /opt/libcorl /opt/libcorl

# install SA sims requirements
# hadolint ignore=DL3013
RUN python --version && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        /opt/libcorl/corl-${CORL_TAG}-py3-none-any.whl

#########################################################################################
# build stage packages the source code
#########################################################################################

FROM develop as build
ENV SA_SIMS_ROOT=/opt/libact3-sa-sims

WORKDIR /opt/project
COPY . .

RUN python setup.py bdist_wheel -d ${SA_SIMS_ROOT}


#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.
FROM develop as cicd
