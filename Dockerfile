# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.1/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt
#################### BASE BUILD IMAGE ####################


#################### EXTENSION BUILD IMAGE ####################
FROM dev AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# copy input files
COPY csrc csrc
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vllm/__init__.py vllm/__init__.py

# cuda arch list used by torch
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads
# make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1

RUN python3 setup.py build_ext --inplace
#################### EXTENSION Build IMAGE ####################


#################### TEST IMAGE ####################
# image to run unit testing suite
FROM dev AS test

# copy pytorch extensions separately to avoid having to rebuild
# when python code changes
WORKDIR /vllm-workspace
# ADD is used to preserve directory structure
ADD . /vllm-workspace/
COPY --from=build /workspace/vllm/*.so /vllm-workspace/vllm/
# ignore build dependencies installation because we are using pre-complied extensions
RUN rm pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip VLLM_USE_PRECOMPILED=1 pip install . --verbose
#################### TEST IMAGE ####################


#################### RUNTIME BASE IMAGE ####################
# We used base cuda image because pytorch installs its own cuda libraries.
# However cupy depends on cuda libraries so we had to switch to the runtime image
# In the future it would be nice to get a container with pytorch and cuda without duplicating cuda
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS vllm-base

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
#################### RUNTIME BASE IMAGE ####################


#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai
# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate

COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY --from=build /workspace/vllm/thirdparty_files /workspace/vllm/thirdparty_files
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
