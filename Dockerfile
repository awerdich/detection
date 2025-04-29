# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.03-py3 AS base

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    NO_COLOR=true \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=true \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr

COPY --from=ghcr.io/astral-sh/uv:0.6.12 /uv /uvx /bin/

# JupyterLab and TensorBoard
EXPOSE 8888
EXPOSE 6006

RUN mkdir -p /app
WORKDIR /app

# Pip and pipenv
RUN pip install --upgrade pip

# Copy the project files to create the environment
COPY uv.lock pyproject.toml README.md .
COPY src/detection/__init__.py src/detection/__init__.py

# Install depenencies that do not rely on the docker dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --inexact

# Install dependencies that use the original docker environment
RUN python -m pip install -U \
    timm \
    accelerate \
    torchmetrics

RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

# Run the jupyter lab server
RUN mkdir -p /run_scripts
COPY /bash_scripts/docker_entry /run_scripts
RUN chmod +x /run_scripts/*
CMD ["/bin/bash", "/run_scripts/docker_entry"]