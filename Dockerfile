# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.01-py3 AS base

ARG DEV_detection

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    PIPENV_HIDE_EMOJIS=true \
    NO_COLOR=true \
    PIPENV_NOSPIN=true

# Jupyter Lab
EXPOSE 8888
# TensorBoard
EXPOSE 6006

RUN mkdir -p /app
WORKDIR /app

# Pip and pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# We need the setup package information
COPY setup.py ./
COPY src/periomodel/__init__.py src/periomodel/__init__.py

# Additional dependencies 
COPY Pipfile Pipfile.lock ./
RUN --mount=source=.git,target=.git,type=bind  \
    pipenv install --system --deploy --ignore-pipfile --dev

RUN python -m pip install -U \
    "numpy<2.0" \
    timm \
    accelerate \
    torchmetrics

RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

# Run the jupyter lab server
RUN mkdir -p /run_scripts
COPY /bash_scripts/docker_entry /run_scripts
RUN chmod +x /run_scripts/*
CMD ["/bin/bash", "/run_scripts/docker_entry"]